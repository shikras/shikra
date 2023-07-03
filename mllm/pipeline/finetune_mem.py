# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import sys
import pathlib
project_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_path))

# Need to call this before importing transformers.
from mllm.utils.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from mllm.pipeline.finetune import main


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
