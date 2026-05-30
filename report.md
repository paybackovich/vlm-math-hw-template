# Report

## Track

Выбранный трек:

```text
A
```

## Что реализовано

- [x] dataset.py
- [x] processor.py
- [x] model.py
- [x] train.py
- [x] benchmark.py

Дополнительно добавлен модуль **`hw/backbones.py`** — фабрика бэкбонов, которая отдаёт лёгкие self-contained mock-энкодер/LLM/токенизатор, что позволяет запускать `train.py`/`benchmark.py` офлайн без скачивания моделей;
для Track B/C — реальные HuggingFace-модели (ViT + Qwen2).

## Конфигурация

```text
config path: configs/track_a_cpu.yaml (train), configs/inference_math.yaml --toy (eval)
seed: 42
device: cpu
dtype: float32
max_steps: 3
batch size: local 1 / global 1 (accumulation = 1)
num_image_tokens: 16, image_size: 224, max_length: 256
```

## Результаты

```text
public tests: 14 passed (pytest -q tests_public)
train loss: step1=4.07, step2=4.95, step3=5.00 (loss конечный; trainable params = 8448 — только adapter)
benchmark accuracy (toy dev): overall ~0.0–0.75 (по 4 примерам, значение нестабильно)
```

## Использованные ресурсы

```text
CPU/GPU: CPU (Apple Silicon, локально), Python 3.12, torch 2.12
VRAM: не использовалась
время обучения: 1.1 c на 3 шага
```

## Анализ ошибок

Поскольку на CPU-треке используется незатренированный mock-LLM, типичные ошибки носят
пайплайновый/baseline-характер:

1. **Случайный выбор варианта.** Модель не выучивает связь «изображение+вопрос → буква»,
   поэтому выбирает вариант фактически случайно (на geometry в одном из прогонов 1.0, на plots — 0.0).
2. **Шумная генерация до парсинга.** `generate` выдаёт несвязный текст из словаря (например, «45°. и диаграмме высоту имеет угол…»), а `parse_mc_answer` достаёт последнюю встретившуюся букву A–E — она часто не совпадает с правильной.
3. **Нет реального reasoning по визуальной части.** Adapter+mock-encoder не извлекают количественную информацию с графика/схемы, поэтому числовые ответы (площадь, гипотенуза, значение столбца) не вычисляются — для качества нужен реальный ViT+LLM.

## Комментарии

Самым неочевидным было то, что шаблон из коробки не запускает команды `train.py`/`benchmark.py`.

## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
