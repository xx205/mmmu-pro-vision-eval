# MMMU-Pro Vision Export & Evaluation

This repo provides three focused utilities:

- export_vision_images_jsonl.py — Export the MMMU-Pro “vision” split: save images to a folder and write a JSONL with per-image metadata.
- export_standard_images_jsonl.py — Export the MMMU-Pro “standard” config: save all related images and write a JSONL per question with images grouped in the order indicated by the problem text (including placeholders), plus question and options (raw and parsed).
- eval_mmmu_pro_vision_imageonly_gpt5.py — Evaluate the vision questions using only the images (no text join). It supports multi-image inputs, RPM limiting, concurrency, and resume by skipping already-finished UUIDs in the results file.

## Requirements

- Python 3.9+
- Packages: openai, pillow, pandas, datasets (only the exporters need `datasets`)

Install:

```
pip install -U openai pillow pandas datasets
```

## 1) Export the vision split

```
python export_vision_images_jsonl.py \
  --split test \
  --out-dir vision_images \
  --jsonl vision_images.jsonl \
  --workers 64
```

Output:
- Images in `vision_images/`
- JSONL `vision_images.jsonl` with fields: `{"image": path, "ori_ans": "Answer: X", "subject": Y, "uuid": Z}`

## 2) Export the standard config (grouped per question)

```
python export_standard_images_jsonl.py \
  --std-config "standard (10 options)" \
  --split test \
  --out-dir standard_images \
  --jsonl standard_images.jsonl \
  --workers 64
```

Each JSONL line (per question) includes:
```
{
  "images": [paths...],        # ordered by placeholders in question/options, then by column order
  "ori_ans": "Answer: X",
  "subject": "...",
  "uuid": "...",
  "question": "...",
  "options_raw_list": [...],
  "options_parsed": [{"label":"A","text":"..."}, ...]
}
```

## 3) Image-only evaluation with GPT‑5

This script evaluates using only the images. The prompt asks the model to read the question and options inside the image(s) and ends with a final line `Answer: X`.

Environment:
- `OPENAI_API_KEY` — API key
- If you use a reverse proxy/compatible gateway, set `BASE_URL` accordingly.

Key features:
- Multi-image per question (sorted by filename image/image_k)
- RPM limiter (global, sliding-window) and `--workers` concurrency
- Resume: reads the existing results JSONL and skips UUIDs already present
- Durable writes: append + flush + fsync per record

Example:
```
export OPENAI_API_KEY=sk-...
python eval_mmmu_pro_vision_imageonly_gpt5.py \
  --jsonl vision_images.jsonl \
  --out results_gpt5_vision.jsonl \
  --rpm 120 --workers 8
```

Result lines include fields:
```
{
  "uuid", "subject", "gold", "pred", "ok",
  "n_images", "model", "response_text", "response", "error"
}
```

Notes:
- The exporter scripts rely on HuggingFace Datasets (MMMU/MMMU_Pro) and Pillow.
- The evaluation script uses the Responses API with text+images inputs.
