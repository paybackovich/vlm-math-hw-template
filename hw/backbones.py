from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from hw.constants import IGNORE_INDEX, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.model import ModelConfig


class SimpleWhitespaceTokenizer:

    def __init__(self) -> None:
        self.id_to_token: list[str] = []
        self.token_to_id: dict[str, int] = {}
        for tok in ("<pad>", "<eos>", "<unk>", IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN):
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def image_token_id(self) -> int:
        return self.token_to_id[IMAGE_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def fit(self, texts: Iterable[str]) -> "SimpleWhitespaceTokenizer":
        for text in texts:
            for token in text.replace("\n", " ").split():
                self._add(token)
        return self

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [
            self.token_to_id.get(tok, self.unk_token_id) for tok in text.replace("\n", " ").split()
        ]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        specials = {
            self.pad_token_id,
            self.eos_token_id,
            self.image_token_id,
            self.token_to_id[IMAGE_START_TOKEN],
            self.token_to_id[IMAGE_END_TOKEN],
        }
        tokens = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in specials:
                continue
            if 0 <= i < len(self.id_to_token):
                tokens.append(self.id_to_token[i])
        return " ".join(tokens)


@dataclass
class _VisionOutput:
    last_hidden_state: torch.Tensor


class TinyVisionEncoder(nn.Module):

    def __init__(self, hidden_size: int = 64, patch_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> _VisionOutput:
        feats = self.patch(pixel_values)  # [N, C, h, w]
        feats = feats.flatten(2).transpose(1, 2)  # [N, P, C]
        return _VisionOutput(last_hidden_state=feats)


@dataclass
class _CausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class TinyLanguageModel(nn.Module):

    _is_mock = True

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Embedding(max_position_embeddings, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> _CausalLMOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        bsz, seq_len, _ = inputs_embeds.shape

        positions = torch.arange(seq_len, device=inputs_embeds.device)
        hidden = inputs_embeds + self.position_embed(positions).unsqueeze(0)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs_embeds.device),
            diagonal=1,
        )
        pad_mask = None
        if attention_mask is not None:
            pad_mask = attention_mask == 0

        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=pad_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return _CausalLMOutput(loss=loss, logits=logits)


def _is_mock_name(name: str | None) -> bool:
    if not name:
        return True
    name = name.lower()
    return any(key in name for key in ("mock", "tiny", "local"))


def build_backbones(
    model_cfg: dict[str, Any],
    num_image_tokens: int,
    corpus: Iterable[str] | None = None,
    force_mock: bool = False,
) -> tuple[nn.Module, nn.Module, Any, ModelConfig]:
    vision_name = model_cfg.get("vision_encoder")
    llm_name = model_cfg.get("language_model")

    if force_mock or _is_mock_name(vision_name) or _is_mock_name(llm_name):
        tokenizer = SimpleWhitespaceTokenizer()
        if corpus is not None:
            tokenizer.fit(corpus)

        vision_hidden = 64
        text_hidden = 64
        vision_encoder = TinyVisionEncoder(hidden_size=vision_hidden, patch_size=16)
        language_model = TinyLanguageModel(vocab_size=tokenizer.vocab_size, hidden_size=text_hidden)
        config = ModelConfig(
            vision_hidden_size=vision_hidden,
            text_hidden_size=text_hidden,
            num_image_tokens=num_image_tokens,
            image_token_id=tokenizer.image_token_id,
        )
        return vision_encoder, language_model, tokenizer, config

    return _build_hf_backbones(vision_name, llm_name, num_image_tokens)


def _build_hf_backbones(
    vision_name: str, llm_name: str, num_image_tokens: int
) -> tuple[nn.Module, nn.Module, Any, ModelConfig]:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    special_tokens = [IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision_encoder = AutoModel.from_pretrained(vision_name)
    language_model = AutoModelForCausalLM.from_pretrained(llm_name)
    language_model.resize_token_embeddings(len(tokenizer))

    config = ModelConfig(
        vision_hidden_size=vision_encoder.config.hidden_size,
        text_hidden_size=language_model.config.hidden_size,
        num_image_tokens=num_image_tokens,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
    )
    return vision_encoder, language_model, tokenizer, config
