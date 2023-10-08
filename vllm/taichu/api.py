import os
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, List, Optional, Tuple, Union
from fastapi import APIRouter  # 导入 APIRouter

# import fastapi
from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.entrypoints.openai.protocol import TaichuRequest, TaichuResponse, TaichuStreamResponse, TaichuErrorResponse

logger = init_logger(__name__)
router = APIRouter()

# Completion-related arguments:
COMPLETION_MAX_PROMPT = int(os.getenv("COMPLETION_MAX_PROMPT", "14000"))
MODEL_PATH = os.environ["MODEL_PATH"]

DEFAULT_PREFIX = (
        "我是由武汉人工智能研究院研发的多模态AI助手，我的名字叫小初。我可以帮你做很多事情哦，例如:\n\n"
        "1. 回答各种类型的问题：历史、科学、文化、娱乐、体育等各种主题的问题，并提供详细的答案和解释。\n"
        "2. 提供建议和信息：根据用户的需求和兴趣，提供各种类型的建议和信息，例如旅游、购物、健康、职业发展等方面的建议。\n"
        "3. 文本创作：生成各种类型的文本，例如诗歌、小说、新闻、电子邮件、商业文案等等。\n"
        "4. 聊天和交流：我可以与用户进行自然而流畅的对话和交流，帮助用户解决问题、获取信息或者进行娱乐。\n"
        "5. 生成图片：我可以根据用户的文字描述，生成符合要求的图片或画作。\n"
        "总之，我可以用自然语言与人类进行交流，并为他们提供各种类型的帮助和服务。无论用户需要什么类型的信息或者帮助，我都会尽力满足他们的需求。\n\n"
        "接下来请开始向我提问吧。\n\n"
    )


def shink_input_size(full_input, max_prompt_size, prefix):
    """ 如果 inputs + full_input 的长度 > max_prompt_size
        对 full_input进行截断，缩减到 max_prompt_size 以内
    """

    header = "{prefix}{instruction}"

    inputs = header.format_map({"instruction": full_input, "prefix": prefix})

    if len(inputs) < max_prompt_size:
        return inputs, full_input
    else:
        logger.warning("[shink_input_size] prompt size: {} large than {}".format(len(inputs), max_prompt_size))
        delta = len(inputs) - max_prompt_size

        full_input_list = full_input.split("###问题")
        full_input_list = [i for i in full_input_list if i != '']
        full_input_list = ["###问题" + i for i in full_input_list]

        delete_sum = 0
        delete_round = 0
        while delete_sum < delta and len(full_input_list) > 1:
            delete_sum += len(full_input_list[0])
            full_input_list.pop(0)
            delete_round += 1

        truncated_full_input = "".join(full_input_list)
        result = header.format_map({"instruction": truncated_full_input,
                                    "prefix": prefix})

        logger.warning("[shink_input_size] prompt size shink to {} after delete {} formal round".format(
            len(result), delete_round))

        return result, truncated_full_input


# from vllm.entrypoints.openai.api_server import engine, tokenizer, max_model_len
from vllm.entrypoints.openai.api_server import create_logprobs, create_error_response, create_taichu_error_response
from vllm.taichu.store.model_store import ModelStore


async def check_length_taichu(
    request: Union[TaichuRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    if prompt_ids is not None:
        input_ids = prompt_ids
    else:
        input_ids = ModelStore().tokenizer(prompt).input_ids
    token_num = len(input_ids)

    # TODO
    # if token_num + request.max_new_tokens > max_model_len:
    if token_num > ModelStore().max_model_len:
        return input_ids, create_taichu_error_response(
            status_code=HTTPStatus.OK, message="输入的文本长度过长，请重新输入",
        )
    else:
        return input_ids, None


# @app.post("/")
@router.post("/")
async def infer(request: TaichuRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - echo (since the vLLM engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """
    logger.info("[infer] Received completion request: {}".format(request))

    # error_check_ret = await check_model(request)
    # if error_check_ret is not None:
    #     return error_check_ret

    if request.echo:
        # We do not support echo since the vLLM engine does not
        # currently support getting the logprobs of prompt tokens.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "echo is not currently supported")

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "suffix is not currently supported")

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    request_id = f"cmpl-{random_uuid()}"

    use_token_ids = False
    # TODO: prompt 预处理
    logger.warning("[infer] input_text: {}".format(request.input_text))
    logger.warning("[infer] input_context: {}".format(request.context))
    logger.warning("[infer] rewrited_input_text: {}".format(request.rewrited_input_text))

    input_text = request.input_text
    context = request.context
    prefix = request.prefix if request.prefix is not None and request.prefix != "" else DEFAULT_PREFIX
    rewrited_input_text = request.rewrited_input_text \
        if request.rewrited_input_text is not None and request.rewrited_input_text != "" else request.input_text

    full_input = context + '\n' + "###问题：\n" + input_text + "\n\n" + "###答案："
    rewrited_full_input = context + '\n' + "###问题：\n" + rewrited_input_text + "\n\n" + "###答案："

    # 如果超长，rewrited_full_input 从头删掉N轮对话
    prompt, truncated_full_input = shink_input_size(
        rewrited_full_input, COMPLETION_MAX_PROMPT, prefix)

    # 判断 token_num + request.max_tokens > max_model_len
    if use_token_ids:
        _, error_check_ret = await check_length_taichu(request, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length_taichu(request, prompt=prompt)

    # 不报错，提示warning
    if error_check_ret is not None:
        logger.warning("[infer] error_check_ret: {}".format(error_check_ret.body))
        return error_check_ret

    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.repetition_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_new_tokens,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if use_token_ids:
        result_generator = ModelStore().engine.generate(None,
                                                        sampling_params,
                                                        request_id,
                                                        prompt_token_ids=prompt)
    else:
        result_generator = ModelStore().engine.generate(prompt, sampling_params, request_id,
                                                        token_ids)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = (request.do_stream
              and (request.best_of is None or request.n == request.best_of)
              and not request.use_beam_search)

    async def abort_request() -> None:
        await ModelStore().engine.abort(request_id)

    def create_stream_response_json(
        index: int,
        text: str,
        logprobs: Optional[LogProbs] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = TaichuStreamResponse(
            id=request_id,
            created=created_time,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n

        generated_text = ''
        generated_index = 0
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                generated_text += delta_text
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        output.token_ids[previous_num_tokens[i]:],
                        output.logprobs[previous_num_tokens[i]:],
                        len(previous_texts[i]))
                else:
                    logprobs = None

                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=generated_index,
                    text=delta_text,
                    logprobs=logprobs,
                )
                yield f"{response_json}\n"

                if output.finish_reason is not None:
                    logprobs = (LogProbs()
                                if request.logprobs is not None else None)
                    response_json = create_stream_response_json(
                        index=generated_index+1,
                        text="",
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                    )
                    yield f"{response_json}\n"

                generated_index += 1
        yield json.dumps({"full_context": full_input + generated_text,
                          'query': input_text,
                          'answer': generated_text,
                          'token_nums': generated_index
                          },
                         ensure_ascii=False)

    # Streaming response
    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=background_tasks)

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        if request.logprobs is not None:
            logprobs = create_logprobs(output.token_ids, output.logprobs)
        else:
            logprobs = None
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output.text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = TaichuResponse(
        id=request_id,
        created=created_time,
        choices=choices,
        usage=usage,
    )

    if request.do_stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response
