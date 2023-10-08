# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from vllm.utils import random_uuid
from vllm.entrypoints.openai.protocol import CompletionResponseStreamChoice, CompletionResponseChoice, UsageInfo


class TaichuErrorResponse(BaseModel):
    message: str
    result: dict = {}
    status: int = 0


class TaichuRequest(BaseModel):
    # model: str
    # a string, array of strings, array of tokens, or array of token arrays
    # prompt: Union[List[int], List[List[int]], str, List[str]]
    input_text: Union[List[int], List[List[int]], str, List[str]]
    context: Union[List[int], List[List[int]], str, List[str]] = ""
    rewrited_input_text: Optional[str] = ""
    prefix: Optional[str] = ""
    suffix: Optional[str] = None
    # max_tokens: Optional[int] = 16
    max_length: Optional[int] = 4096
    max_new_tokens: Optional[int] = 2048
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    do_stream: Optional[bool] = True
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    # presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by vLLM
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False


class TaichuStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "taichu_infer"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[CompletionResponseStreamChoice]


class TaichuResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "taichu_infer"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
