from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class T5Engine:
    def __init__(self, model_name: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, texts: str | Sequence[str], max_length: int = 256):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        batch = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        encoder = self.model.get_encoder()
        outputs = encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        hidden = outputs.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        vectors = pooled.detach().cpu().numpy().astype(np.float32)
        return vectors[0] if single else vectors

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 180,
        num_beams: int = 4,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()
