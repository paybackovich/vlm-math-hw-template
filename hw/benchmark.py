from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    if not text:
        return None
    pattern = r"\b([" + "".join(choices) + r"])\b"
    matches = re.findall(pattern, text.upper())
    if not matches:
        return None
    return matches[-1]


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """
    import torch

    from hw.backbones import build_backbones
    from hw.dataset import MathVQADataset
    from hw.model import MathVLM
    from hw.processor import MathVLMProcessor, ProcessorConfig

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    proc_cfg = config.get("processor", {})
    infer_cfg = config.get("inference", {})

    manifest = data_cfg.get("eval_manifest") or data_cfg.get("train_manifest")
    split = data_cfg.get("split", "dev")
    max_samples = data_cfg.get("max_samples")
    if toy:
        max_samples = max_samples or 8

    dataset = MathVQADataset(manifest, split=split, max_samples=max_samples)

    device = torch.device(infer_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    processor_config = ProcessorConfig(
        image_size=int(proc_cfg.get("image_size", 224)),
        num_tiles=int(proc_cfg.get("num_tiles", 1)),
        num_image_tokens=int(proc_cfg.get("num_image_tokens", 49)),
        max_length=int(proc_cfg.get("max_length", 512)),
        ignore_index=int(proc_cfg.get("ignore_index", -100)),
    )

    corpus = []
    for i in range(len(dataset)):
        s = dataset[i]
        corpus.append(s.question)
        corpus.extend(s.options)
        corpus.append(s.answer)

    vision_encoder, language_model, tokenizer, model_config = build_backbones(
        model_cfg,
        num_image_tokens=processor_config.num_image_tokens,
        corpus=corpus,
        force_mock=toy,
    )
    processor = MathVLMProcessor(tokenizer, processor_config)

    model = MathVLM(vision_encoder, language_model, model_config)
    model.freeze_backbones()

    adapter_path = model_cfg.get("adapter_path")
    if adapter_path and Path(adapter_path).exists():
        if str(adapter_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(str(adapter_path))
        else:
            state_dict = torch.load(str(adapter_path), map_location="cpu")
        model.adapter.load_state_dict(state_dict)
        print(f"loaded adapter from {adapter_path}")

    model.to(device)
    model.eval()

    max_new_tokens = int(infer_cfg.get("max_new_tokens", 16))
    rows: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        prompt = processor.build_prompt(sample, include_answer=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[
            : processor_config.max_length
        ]
        input_ids = torch.tensor(prompt_ids, dtype=torch.long)
        batch = {
            "input_ids": input_ids.unsqueeze(0).to(device),
            "attention_mask": torch.ones_like(input_ids).unsqueeze(0).to(device),
            "pixel_values": processor.preprocess_image(sample.image).unsqueeze(0).to(device),
        }
        gen_ids = model.generate(
            batch, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(gen_ids[0].tolist())
        prediction = parse_mc_answer(generated_text)
        rows.append(
            {
                "id": sample.id,
                "prediction": prediction,
                "answer": sample.answer,
                "subject": sample.subject,
                "generated": generated_text,
            }
        )

    output_path = infer_cfg.get("output_path")
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} predictions to {output_path}")

    metrics = compute_accuracy(rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
