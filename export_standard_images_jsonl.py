#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 MMMU/MMMU_Pro 的 standard 配置导出为：
  - 图片：保存到目录 `standard_images/`
  - 信息：按题聚合写入 `standard_images.jsonl`，字段为：
        {"images": [image_path, ...], "ori_ans": "Answer: X", "subject": "Y", "uuid": "Z",
         "question": "...", "options_raw_list": [..], "options_parsed": [{"label":"A","text":"..."}, ...]}

与 vision 相比，standard 更常见：
  - 单题可包含多张图片（image_1, image_2, ...）
  - 题干/选项可能包含 <image k> 占位；本脚本不解析锚点，只输出题干与选项（原始与解析后）。
  - 多图题按列顺序聚合到同一行的 `images` 列表中。

如需导出锚点映射、题干、选项、规范化答案等，可在此基础上扩展（见脚本末注释）。

依赖：pip install -U datasets pillow pandas

示例：
  python export_standard_images_jsonl.py \
    --std-config "standard (10 options)" \
    --split test \
    --out-dir standard_images \
    --jsonl standard_images.jsonl \
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


# --------------------- Helpers (shared pattern) ---------------------
def pick_image_columns(colnames: Sequence[str]) -> List[str]:
    """返回图片列名，保证顺序：`image` 在前，后跟 `image_1..image_n` 按数值升序。"""
    cols: List[str] = []
    if "image" in colnames:
        cols.append("image")
    pat = re.compile(r"^image_(\d+)$", re.IGNORECASE)
    numbered: List[Tuple[int, str]] = []
    for c in colnames:
        m = pat.match(c)
        if m:
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            numbered.append((idx, c))
    for _, c in sorted(numbered, key=lambda x: x[0]):
        cols.append(c)
    # 去重保持顺序
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
    return task[0], str(dst.as_posix()), w, h


def normalize_options(options_field: Any) -> List[str]:
    """统一 options -> list[str]；兼容字符串化列表。"""
    if isinstance(options_field, list):
        return ["" if x is None else str(x) for x in options_field]
    if isinstance(options_field, str):
        s = options_field.strip()
        # 尝试按 Python 字面量解析
        import ast

        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return ["" if x is None else str(x) for x in v]
        except Exception:
            pass
        # 回退：抽取引号内片段
        pattern = re.compile(r"""
            '([^']*)'   |   "([^"]*)"
        """, re.VERBOSE)
        parts = pattern.findall(s)
        flat: List[str] = []
        for a, b in parts:
            t = a if a else b
            if t != "":
                flat.append(t)
        return flat if flat else [s]
    return []


def extract_lettered_options_from_question(text: Any) -> Dict[str, str]:
    """从题干中解析 'A./A)/A:/' 等样式的选项文本，返回 {letter: text}。仅 A..J。"""
    if not isinstance(text, str) or not text.strip():
        return {}
    s = text
    pat = re.compile(
        r"(?:^|[\s\[{(,，;；])(?:\(|\[)?([A-J])(?:\)|\])?\s*(?:[.．:：、)])\s*",
        flags=re.MULTILINE,
    )
    matches = list(pat.finditer(s))
    if not matches:
        return {}
    out: Dict[str, str] = {}
    for i, m in enumerate(matches):
        L = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        seg = s[start:end].strip()
        seg = re.sub(r"^[\s\-–—:：、.．。)*）]*", "", seg)
        seg = seg.strip()
        if L not in out:
            out[L] = seg
    return out


def parse_placeholders(text: Any) -> List[int]:
    """抽取文本中的 <image k> 序列（按出现顺序，保留重复）。"""
    if not isinstance(text, str):
        return []
    return [int(m.group(1)) for m in re.finditer(r"<\s*image\s*(\d+)\s*>", text, flags=re.IGNORECASE)]


def placeholders_from_options(opts: List[str]) -> List[int]:
    ks: List[int] = []
    for t in opts:
        m = re.fullmatch(r"\s*<\s*image\s*(\d+)\s*>\s*", t, flags=re.IGNORECASE)
        if m:
            ks.append(int(m.group(1)))
    return ks


# --------------------- Main ---------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std-config", default="standard (10 options)", help="MMMU_Pro standard 配置名")
    ap.add_argument("--split", default="test", help="split 名称，例如 train/validation/test")
    ap.add_argument("--subject", default=None, help="仅导出指定 subject（可选）")
    ap.add_argument("--out-dir", default="standard_images", help="图片输出目录")
    ap.add_argument("--jsonl", default="standard_images.jsonl", help="信息输出 JSONL 文件名")
    # 多进程与 JPEG 选项
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 8))
    ap.add_argument("--chunksize", type=int, default=16)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--jpeg-optimize", action="store_true")
    ap.add_argument("--jpeg-progressive", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 standard 集合
    ds_all = load_dataset("MMMU/MMMU_Pro", args.std_config)
    assert args.split in ds_all, f"Split not found. Avail={list(ds_all.keys())}"
    ds = ds_all[args.split]

    # 2) 将图像列设为 decode=False
    img_cols = pick_image_columns(ds.column_names)
    for c in img_cols:
        ds = ds.cast_column(c, HFImage(decode=False))

    # 3) 过滤 + 转 pandas
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

    keep = ["id", "uuid", "subject", "answer", "question", "options"] + img_cols
    df = to_pd(ds, args.subject, keep, img_cols)
    if df.empty:
        print("[!] No rows after filtering; nothing to export.")
        return

    # 4) 组装保存任务（并按题聚合元信息）
    tasks: List[Tuple[str, str, bytes, str, int, bool, bool]] = []
    # 每题元信息（保持插入顺序）：rid -> {..., image_keys:[base_key,...]}
    meta_by_rid: Dict[str, Dict[str, Any]] = {}
    # base_key -> rid，用于回填
    bkey_to_rid: Dict[str, str] = {}

    letters = list("ABCDEFGHIJ")

    def add_tasks_for_row(row: pd.Series):
        rid = str(row.get("uuid") or row.get("id"))
        subj = row.get("subject", "")
        ans = row.get("answer", "")
        question = row.get("question", "")
        # 归一化原始 options 为列表
        options_raw_list = normalize_options(row.get("options", []))
        cleaned = [("" if x is None else str(x)).strip() for x in options_raw_list]
        # 解析后的 options：
        def _as_letter_token(s: str) -> bool:
            t = re.sub(r"[.)\s]", "", s or "")
            return bool(re.fullmatch(r"[A-J]", t, flags=re.IGNORECASE))

        is_pure_letters = len(cleaned) > 0 and all(_as_letter_token(x) for x in cleaned)
        options_parsed: List[Dict[str, str]] = []
        if is_pure_letters:
            parsed_from_q = extract_lettered_options_from_question(question)
            for i in range(len(cleaned)):
                L = letters[i] if i < len(letters) else "?"
                txt = parsed_from_q.get(L, "") or cleaned[i]
                options_parsed.append({"label": L, "text": txt})
        else:
            for i, t in enumerate(cleaned):
                L = letters[i] if i < len(letters) else "?"
                options_parsed.append({"label": L, "text": t})
        # 初始化题级元数据（若首次出现）
        if rid not in meta_by_rid:
            meta_by_rid[rid] = {
                "uuid": rid,
                "subject": subj,
                "ori_ans": f"Answer: {'' if ans is None else str(ans)}",
                "question": question,
                "options_raw_list": cleaned,
                "options_parsed": options_parsed,
                "image_keys": [],  # 顺序收集 base_key（列顺序）
                "k_to_bkey": {},   # k -> base_key，用于按题面占位排序
            }

        for col in (row.get("_img_cols") or []):
            if col not in row:
                continue
            cell = row[col]
            base_key = f"{rid}_{col}"
            # 解析列号 k：image -> 1；image_N -> N
            k = 1
            m = re.match(r"^image_(\d+)$", str(col), flags=re.IGNORECASE)
            if m:
                try:
                    k = int(m.group(1))
                except Exception:
                    k = 1
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
            meta_by_rid[rid]["image_keys"].append(base_key)
            bkey_to_rid[base_key] = rid
            # 记录 k -> base_key（首次为准）
            if k not in meta_by_rid[rid]["k_to_bkey"]:
                meta_by_rid[rid]["k_to_bkey"][k] = base_key

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

    # 6) 写 JSONL（按题聚合一行，images 按列顺序）
    n_lines = 0
    jsonl_path = Path(args.jsonl)
    import json

    with jsonl_path.open("w", encoding="utf-8") as fw:
        for rid, meta in meta_by_rid.items():
            keys_in_order: List[str] = meta.get("image_keys", [])
            # 依据题面占位 <image k> 的顺序重排
            q = meta.get("question", "")
            opts_raw: List[str] = meta.get("options_raw_list", [])
            occ_q = parse_placeholders(q)
            occ_opts = placeholders_from_options(opts_raw)
            occ_seq = occ_q if len(occ_q) > 0 else occ_opts
            # 构建 k -> path
            k_to_bkey: Dict[int, str] = meta.get("k_to_bkey", {})
            used: set = set()
            images: List[str] = []
            # 先按出现顺序添加
            for k in occ_seq:
                bkey = k_to_bkey.get(k)
                if bkey and bkey in results and bkey not in used:
                    images.append(results[bkey][0])
                    used.add(bkey)
            # 再补充剩余列（按列顺序）
            for bkey in keys_in_order:
                if bkey in results and bkey not in used:
                    images.append(results[bkey][0])
                    used.add(bkey)
            line = {
                "images": images,
                "ori_ans": meta.get("ori_ans", "Answer: "),
                "subject": meta.get("subject", ""),
                "uuid": meta.get("uuid", ""),
                "question": meta.get("question", ""),
                "options_raw_list": meta.get("options_raw_list", []),
                "options_parsed": meta.get("options_parsed", []),
            }
            fw.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_lines += 1
    print(f"[✓] Wrote JSONL: {jsonl_path.resolve()} ({n_lines} lines, aggregated)")

    # 扩展示例（未实现）：
    # - 增加字段：question, options_raw, explanation, img_type, topic_difficulty
    # - 提供占位映射：{"k_to_path": {1: ".../image_1.jpg", ...}, "occ_seq": [按题干/选项出现顺序的 k 列表]}
    # - 归一化答案字母/文本，便于后续评测或可视化


if __name__ == "__main__":
    if Image is None:
        raise SystemExit("Pillow 未安装或导入失败：请先 `pip install pillow`")
    main()
