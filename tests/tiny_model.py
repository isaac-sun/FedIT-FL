"""A tiny random Qwen2 model and BPE tokenizer for CPU tests (no download)."""

from __future__ import annotations

import os

from data.dolly import load_records

EOS = "<|endoftext|>"


def build_tiny_model(folder: str, data_path: str, vocab_size: int = 600) -> str:
    if os.path.exists(os.path.join(folder, "model.safetensors")):
        return folder
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    frame = load_records(data_path, check_sha256=False).head(3000)
    texts = (frame["instruction"] + " " + frame["context"] + " " + frame["response"]).tolist()
    texts.append("### Instruction: ### Input: ### Response: Below is an instruction")
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=EOS, pad_token=EOS)
    os.makedirs(folder, exist_ok=True)
    fast.save_pretrained(folder)
    config = Qwen2Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=1024, tie_word_embeddings=True,
                         eos_token_id=fast.eos_token_id, bos_token_id=fast.eos_token_id)
    Qwen2ForCausalLM(config).save_pretrained(folder, safe_serialization=True)
    return folder
