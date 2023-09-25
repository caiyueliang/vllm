"""Utilities for selecting and loading models."""
import contextlib
import logging
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import *  # pylint: disable=wildcard-import
from vllm.model_executor.weight_utils import initialize_dummy_weights

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "AquilaModel": AquilaForCausalLM,
    "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    "BloomForCausalLM": BloomForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "RWForCausalLM": FalconForCausalLM,
}


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        model = model_class(model_config.hf_config)
        if model_config.load_format == "dummy":
            model = model.cuda()
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format)
            model = model.cuda()
    return model.eval()


from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_new(name_or_path,
                 torch_dtype=torch.float16,
                 use_flash_attn=False,
                 max_input_len=2048,
                 device_map='auto'):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_num = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logging.warning("The number of gpus is {}".format(device_num))

        model_init_kwargs = dict(
            pretrained_model_name_or_path=name_or_path,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        assert torch_dtype != torch.bfloat16, ("bfloat16 will cause error when use RoPE or "
                                               "Alibi (Refer:https://mp.weixin.qq.com/s/qA6rdFUPmPsd4elxGnNf2A) ")
        if torch_dtype == torch.float16:
            logging.warning(("half precision may cause error when RoPE or Alibi is used "
                             "(Refer:https://mp.weixin.qq.com/s/qA6rdFUPmPsd4elxGnNf2A)"))
        if use_flash_attn:
            model_init_kwargs['use_flash_attn'] = True

        logging.warning("[get_model_new] model_init_kwargs: {}".format(model_init_kwargs))
        model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)

        # tokenizer = AutoTokenizer.from_pretrained(name_or_path,
        #                                                use_fast=False,
        #                                                trust_remote_code=True,
        #                                                truncation_side='left',
        #                                                model_max_length=max_input_len)

        return model
