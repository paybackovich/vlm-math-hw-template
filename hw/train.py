from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    model.train()
    optimizer.zero_grad()

    output = model(batch)
    loss = output["loss"] if isinstance(output, dict) else output.loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")

    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    # Imported here to keep module import cheap and avoid heavy deps at import time.
    from hw.backbones import build_backbones
    from hw.dataset import MathVQADataset
    from hw.model import MathVLM
    from hw.processor import MathVLMProcessor, ProcessorConfig

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    proc_cfg = config.get("processor", {})
    trainer_cfg = config.get("trainer", {})

    device = torch.device(trainer_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    max_samples = data_cfg.get("max_samples")
    if fast_train:
        max_samples = min(4, max_samples) if max_samples else 4

    dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=max_samples,
    )

    processor_config = ProcessorConfig(
        image_size=int(proc_cfg.get("image_size", 224)),
        num_tiles=int(proc_cfg.get("num_tiles", 1)),
        tile_overlap=float(proc_cfg.get("tile_overlap", 0.0)),
        num_image_tokens=int(proc_cfg.get("num_image_tokens", 49)),
        max_length=int(proc_cfg.get("max_length", 512)),
        ignore_index=int(proc_cfg.get("ignore_index", -100)),
    )

    # Build a corpus so the mock tokenizer covers everything it will encounter.
    corpus = []
    for i in range(len(dataset)):
        s = dataset[i]
        corpus.append(s.question)
        corpus.extend(s.options)
        corpus.append(s.answer)

    vision_encoder, language_model, tokenizer, model_config = build_backbones(
        model_cfg, num_image_tokens=processor_config.num_image_tokens, corpus=corpus
    )
    processor = MathVLMProcessor(tokenizer, processor_config)

    model = MathVLM(vision_encoder, language_model, model_config)
    if model_cfg.get("freeze_vision", True) or model_cfg.get("freeze_llm", True):
        model.freeze_backbones()
    model.to(device)

    def collate_samples(samples: list[Any]) -> dict[str, torch.Tensor]:
        return processor.collate([processor(s) for s in samples])

    batch_size = int(trainer_cfg.get("local_batch_size", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(trainer_cfg.get("num_workers", 0)),
        collate_fn=collate_samples,
        drop_last=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(trainer_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    local_bs = int(trainer_cfg.get("local_batch_size", 1))
    global_bs = int(trainer_cfg.get("global_batch_size", local_bs))
    accum_steps = max(1, global_bs // max(1, local_bs))

    max_steps = int(trainer_cfg.get("max_steps", 3))
    if fast_train:
        max_steps = min(max_steps, 2)
    num_epochs = int(trainer_cfg.get("num_train_epochs", 1))

    model.train()
    step = 0
    losses: list[float] = []
    micro = 0
    optimizer.zero_grad()
    stop = False
    for _ in range(num_epochs):
        if stop:
            break
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch)
            loss = output["loss"] if isinstance(output, dict) else output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

            (loss / accum_steps).backward()
            micro += 1

            if micro % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                losses.append(float(loss.detach().item()))
                print(f"step {step}/{max_steps} loss={losses[-1]:.4f}")
                if step >= max_steps:
                    stop = True
                    break

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = model.adapter.state_dict()
        if save_path.suffix == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, str(save_path))
        else:
            torch.save(state_dict, str(save_path))
        print(f"saved adapter to {save_path}")

    summary = {
        "steps": step,
        "final_loss": losses[-1] if losses else None,
        "device": str(device),
        "trainable_params": sum(p.numel() for p in trainable),
    }
    print(f"training summary: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
