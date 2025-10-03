#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 MMMU/MMMU_Pro 的 vision 集合导出为：
  - 图片：保存到目录 `vision_images/`
  - 信息：逐图写入 `vision_images.jsonl`，字段为：
        {"image": image_path, "ori_ans": "Answer: X", "subject": "Y", "uuid": "Z"}

参考当前目录下的 visualize_mmmu_pro_pairs_idjoin_mathjax_mp_optfix.py 的读数与落盘方式：
  - 使用 datasets.load_dataset 加载 `vision` 配置
  - 通过 cast_column(Image(decode=False)) 获取 bytes/path
  - 依据是否有 alpha 决定 JPEG/PNG 保存
  - 多进程保存原图

依赖：pip install -U datasets pillow pandas

示例：
  python export_vision_images_jsonl.py \
    --split test \
    --out-dir vision_images \
    --jsonl vision_images.jsonl \
    --workers 64
"""
import argparse
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from datasets import Image as HFImage, load_dataset

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore


# --------------------- Helpers (borrowed/adapted) ---------------------
def pick_image_columns(colnames: Sequence[str]) -> List[str]:
    cols: List[str] = []
    if "image" in colnames:
        cols.append("image")
    pat = re.compile(r"^image_\d+$", re.IGNORECASE)
    cols.extend([c for c in colnames if pat.match(c)])
    # 去重并保持顺序
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _decide_format(im: "Image.Image") -> Tuple[str, str]:
    has_alpha = (im.mode in ("RGBA", "LA")) or (im.mode == "P" and "transparency" in im.info)
    return ("PNG", ".png") if has_alpha else ("JPEG", ".jpg")


def worker_save_original(task: Tuple[str, str, bytes, str, int, bool, bool]) -> Tuple[str, str, int, int]:
    """子进程：将 bytes/path 的原图写盘为 JPEG/PNG（保留原尺寸）。

    返回：(base_key, rel_path, width, height)
    rel_path 为相对当前工作目录的路径字符串。
    """
    (base_key, out_dir, img_bytes, img_path, jpeg_quality, jpeg_optimize, jpeg_progressive) = task
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    im = Image.open(BytesIO(img_bytes)) if img_bytes else Image.open(img_path)
    fmt, suffix = _decide_format(im)
    im = im.convert("RGB") if fmt == "JPEG" else im.convert("RGBA")
    w, h = im.size
    dst = out_dir_p / f"{base_key}{suffix}"
    try:
        if fmt == "JPEG":
            im.save(dst, fmt, quality=jpeg_quality, optimize=jpeg_optimize, progressive=jpeg_progressive)
        else:
            im.save(dst, fmt, optimize=True)
    except Exception:
        dst = out_dir_p / f"{base_key}.png"
        im.convert("RGBA").save(dst, "PNG", optimize=True)
    # 返回相对于 CWD 的路径
    rel_path = str(dst.as_posix())
    return base_key, rel_path, w, h


# --------------------- Main ---------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="vision split 名称，例如 train/validation/test")
    ap.add_argument("--subject", default=None, help="仅导出指定 subject（可选）")
    ap.add_argument("--out-dir", default="vision_images", help="图片输出目录")
    ap.add_argument("--jsonl", default="vision_images.jsonl", help="信息输出 JSONL 文件名")
    # 多进程与 JPEG 选项
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 8))
    ap.add_argument("--chunksize", type=int, default=16)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--jpeg-optimize", action="store_true")
    ap.add_argument("--jpeg-progressive", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 vision 集合
    ds_all = load_dataset("MMMU/MMMU_Pro", "vision")
    assert args.split in ds_all, f"Split not found. Avail={list(ds_all.keys())}"
    ds = ds_all[args.split]

    # 2) 将图像列设为 decode=False 以便拿到 bytes/path
    img_cols = pick_image_columns(ds.column_names)
    for c in img_cols:
        ds = ds.cast_column(c, HFImage(decode=False))

    # 3) 过滤 + 转 pandas 以便遍历
    def to_pd(ds, subject: Optional[str], keep_cols: Sequence[str], img_cols: Sequence[str]):
        rows = []
        for i in range(len(ds)):
            row = ds[i]
            if subject and str(row.get("subject", "")) != subject:
                continue
            rows.append(row)
        df = pd.DataFrame(rows)
        for k in keep_cols:
            if k not in df.columns:
                df[k] = None
        df["_img_cols"] = [list(img_cols)] * len(df)
        return df

    keep = ["id", "uuid", "subject", "answer"] + img_cols
    df = to_pd(ds, args.subject, keep, img_cols)
    if df.empty:
        print("[!] No rows after filtering; nothing to export.")
        return

    # 4) 组装保存任务
    tasks: List[Tuple[str, str, bytes, str, int, bool, bool]] = []
    meta: Dict[str, Dict[str, Any]] = {}

    def add_tasks_for_row(row: pd.Series):
        rid = str(row.get("uuid") or row.get("id"))
        subj = row.get("subject", "")
        ans = row.get("answer", "")
        for col in (row.get("_img_cols") or []):
            if col not in row:
                continue
            cell = row[col]
            base_key = f"{rid}_{col}"
            img_bytes, img_path = b"", ""
            if isinstance(cell, dict):
                img_bytes = cell.get("bytes") or b""
                img_path = cell.get("path") or ""
            elif isinstance(cell, str):
                img_path = cell
            else:
                continue
            tasks.append(
                (
                    base_key,
                    str(out_dir),
                    img_bytes,
                    img_path,
                    int(args.jpeg_quality),
                    bool(args.jpeg_optimize),
                    bool(args.jpeg_progressive),
                )
            )
            meta[base_key] = {
                "uuid": rid,
                "subject": subj,
                "ori_ans": f"Answer: {'' if ans is None else str(ans)}",
            }

    for _, row in df.iterrows():
        add_tasks_for_row(row)

    # 5) 多进程写盘
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: Dict[str, Tuple[str, int, int]] = {}
    if tasks:
        print(f"[i] Saving {len(tasks)} images to '{out_dir}' with {args.workers} processes ...")
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = [ex.submit(worker_save_original, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                base_key, rel, w, h = f.result()
                results[base_key] = (rel, w, h)
                if i % 200 == 0:
                    print(f"  [..] {i}/{len(tasks)}")
        print(f"[✓] Images saved: {len(results)}")
    else:
        print("[i] No images to save.")

    # 6) 写 JSONL（逐图一行）
    n_lines = 0
    jsonl_path = Path(args.jsonl)
    with jsonl_path.open("w", encoding="utf-8") as fw:
        for base_key, (rel, _w, _h) in results.items():
            m = meta.get(base_key, {})
            line = {
                "image": rel,
                "ori_ans": m.get("ori_ans", "Answer: "),
                "subject": m.get("subject", ""),
                "uuid": m.get("uuid", ""),
            }
            import json

            fw.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_lines += 1
    print(f"[✓] Wrote JSONL: {jsonl_path.resolve()} ({n_lines} lines)")


if __name__ == "__main__":
    if Image is None:
        raise SystemExit("Pillow 未安装或导入失败：请先 `pip install pillow`")
    main()

