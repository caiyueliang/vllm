import logging

logger = logging.getLogger(__name__)


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
