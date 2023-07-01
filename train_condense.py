# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.


import sys

if __name__ == "__main__":
    from longchat.train.monkey_patch.llama_condense_monkey_patch import replace_llama_with_condense

    condense_ratio = int(sys.argv[1])
    print(f"Condense Ratio: {condense_ratio}")
    replace_llama_with_condense(ratio=condense_ratio)

    from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

    replace_llama_attn_with_flash_attn()

    from train import train

    train(sys.argv[2:])
