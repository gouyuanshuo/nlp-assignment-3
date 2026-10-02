"""Frozen local BERT with explicit wordpiece-to-word pooling."""

from pathlib import Path

import numpy as np
import torch


class BertWordEncoder:
    def __init__(self, model_path: Path, device: str = "cpu", batch_size: int = 32):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer with word IDs is required")
        self.model = (
            AutoModel.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.tokenizer.model_max_length = min(
            self.tokenizer.model_max_length, self.model.config.max_position_embeddings
        )
        self.device, self.batch_size = device, batch_size

    @torch.inference_mode()
    def encode(self, windows) -> list[np.ndarray]:
        result = []
        for start in range(0, len(windows), self.batch_size):
            chunk = windows[start : start + self.batch_size]
            encoded = self.tokenizer(
                [[w.text for w in words] for words in chunk],
                is_split_into_words=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if encoded["input_ids"].shape[1] > self.tokenizer.model_max_length:
                raise ValueError("Window exceeds the BERT token limit")
            hidden = self.model(
                **{k: v.to(self.device) for k, v in encoded.items()}
            ).last_hidden_state
            for i, words in enumerate(chunk):
                ids = encoded.word_ids(batch_index=i)
                vectors = []
                for word_index in range(len(words)):
                    positions = [j for j, index in enumerate(ids) if index == word_index]
                    if not positions:
                        raise ValueError("Tokenizer omitted a transcript word")
                    vectors.append(hidden[i, positions].mean(dim=0))
                result.append(torch.stack(vectors).float().cpu().numpy())
        return result
