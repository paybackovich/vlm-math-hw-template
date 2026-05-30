from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.benchmark import build_benchmark_prompt
from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample

# ImageNet-style normalization constants.
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        image = image.convert("RGB")
        size = self.config.image_size
        num_tiles = max(1, int(self.config.num_tiles))

        if num_tiles == 1:
            tiles = [image]
        else:
            rows = max(1, int(math.floor(math.sqrt(num_tiles))))
            while num_tiles % rows != 0:
                rows -= 1
            cols = num_tiles // rows
            big = image.resize((cols * size, rows * size), Image.BILINEAR)
            tiles = [
                big.crop((c * size, r * size, (c + 1) * size, (r + 1) * size))
                for r in range(rows)
                for c in range(cols)
            ]

        mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGE_STD).view(3, 1, 1)
        pixel_values = []
        for tile in tiles:
            tile = tile.resize((size, size), Image.BILINEAR)
            tensor = torch.frombuffer(bytearray(tile.tobytes()), dtype=torch.uint8)
            tensor = tensor.view(size, size, 3).permute(2, 0, 1).float() / 255.0
            pixel_values.append((tensor - mean) / std)
        return torch.stack(pixel_values, dim=0)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        placeholder = f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}"
        text_prompt = build_benchmark_prompt(sample.question, sample.options)
        prompt = f"{placeholder}\n{text_prompt}"
        if include_answer:
            prompt = f"{prompt} {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt_text = self.build_prompt(sample, include_answer=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        answer_text = f" {sample.answer}"
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            answer_ids = answer_ids + [eos_id]

        input_ids = prompt_ids + answer_ids
        labels = [self.config.ignore_index] * len(prompt_ids) + list(answer_ids)

        max_length = self.config.max_length
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            length = item["input_ids"].size(0)
            pad = max_len - length
            input_ids.append(
                torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            labels.append(
                torch.cat(
                    [item["labels"], torch.full((pad,), self.config.ignore_index, dtype=torch.long)]
                )
            )

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.stack([item["pixel_values"] for item in batch], dim=0)
        return out
