from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.norm = nn.LayerNorm(vision_hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        x = self.norm(vision_hidden_states)
        x = self.proj(x)  # [B, L, text_hidden_size]

        if x.size(1) != self.num_image_tokens:
            x = x.transpose(1, 2)  # [B, D, L]
            x = F.adaptive_avg_pool1d(x, self.num_image_tokens)  # [B, D, K]
            x = x.transpose(1, 2)  # [B, K, D]
        return x


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    merged = input_embeds.clone()
    mask = input_ids == image_token_id  # [B, L]
    flat_visual = visual_embeds.reshape(-1, visual_embeds.size(-1)).to(
        dtype=merged.dtype, device=merged.device
    )
    merged[mask] = flat_visual
    return merged


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss.

        TODO:
            - encode images;
            - map to visual embeddings;
            - get text input embeddings;
            - merge visual/text embeddings;
            - call language_model with inputs_embeds, attention_mask, labels.
        """
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]
        b, t = pixel_values.shape[:2]

        vision_out = self.vision_encoder(pixel_values.flatten(0, 1))
        hidden = getattr(vision_out, "last_hidden_state", vision_out)
        hidden = hidden.reshape(b, t * hidden.size(1), hidden.size(-1))
        visual_embeds = self.adapter(hidden)

        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = merge_visual_embeddings(
            text_embeds, input_ids, visual_embeds, self.config.image_token_id
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        max_new_tokens = int(generation_kwargs.pop("max_new_tokens", 16))
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        pixel_values = batch["pixel_values"]
        b, t = pixel_values.shape[:2]

        vision_out = self.vision_encoder(pixel_values.flatten(0, 1))
        hidden = getattr(vision_out, "last_hidden_state", vision_out)
        hidden = hidden.reshape(b, t * hidden.size(1), hidden.size(-1))
        visual_embeds = self.adapter(hidden)

        embed_layer = self.language_model.get_input_embeddings()
        inputs_embeds = merge_visual_embeddings(
            embed_layer(input_ids), input_ids, visual_embeds, self.config.image_token_id
        )

        if not getattr(self.language_model, "_is_mock", False) and hasattr(
            self.language_model, "generate"
        ):
            return self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )

        # Manual greedy decoding for the lightweight mock language model.
        cur_embeds = inputs_embeds
        cur_mask = attention_mask
        eos_id = generation_kwargs.get("eos_token_id")
        generated: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            out = self.language_model(inputs_embeds=cur_embeds, attention_mask=cur_mask)
            logits = out.logits if hasattr(out, "logits") else out["logits"]
            next_id = logits[:, -1, :].argmax(dim=-1)  # [B]
            generated.append(next_id)
            next_embed = embed_layer(next_id).unsqueeze(1)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask, torch.ones_like(next_id).unsqueeze(1)], dim=1
                )
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return torch.stack(generated, dim=1)  # [B, T_new]
