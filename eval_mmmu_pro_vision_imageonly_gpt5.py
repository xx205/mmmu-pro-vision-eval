#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMMU-Pro vision 集合评测（纯图，使用 vision_images.jsonl 输入，GPT‑5 High）

思路
- 只依赖 vision_images.jsonl：从中按 uuid 聚合多图并读取金标准答案（"ori_ans": "Answer: X"）。
- 一题可多图：依据文件名中的 image/image_k 顺序聚合后一起发送给模型。
- 提示词：告知题目与选项均在图内，要求输出 'Final Answer: X'（X∈A..J）。

依赖
- pip install -U openai pillow pandas

示例
  export OPENAI_API_KEY=sk-xxxx
  python eval_mmmu_pro_vision_imageonly_gpt5.py \
    --jsonl vision_images.jsonl --out results_gpt5_vision.jsonl --limit 500
"""
import argparse
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# --------------------- Helpers ---------------------
def parse_answer_letter(s: str) -> str:
    """解析模型输出中“最后一个 Answer: X”的 X（A..J，忽略大小写）。

    规则：
    - 优先匹配所有 (Final )?Answer: <LETTER>，返回最后一个匹配的字母；
    - 若未出现“Answer:”，回退为寻找末行单字母或全文中的第一个 A..J。
    """
    if not s:
        return ""
    matches = re.findall(r"(?i)(?:final\s+)?answer\s*[:：]\s*([A-J])\b", s)
    if matches:
        return matches[-1].upper()
    # 回退策略（鲁棒性）：末行单字母 -> 全文第一个 A..J
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.fullmatch(r"[A-J]", ln, flags=re.IGNORECASE):
            return ln.upper()
    m2 = re.search(r"\b([A-J])\b", s, flags=re.IGNORECASE)
    return m2.group(1).upper() if m2 else ""


def letter_from_ori_ans(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = re.search(r"(?i)answer\s*[:：]\s*([A-J])\b", s.strip())
    return (m.group(1) if m else s).strip().upper()


def encode_image_to_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = mimetypes.types_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_prompt_image_only() -> str:
    return "\n".join([
        # 不提数据集名，指示从图中读取题面与选项
        "Solve the multiple‑choice visual question using only the provided image(s).",
        "All necessary information (question and options A‑J) appears inside the images.",
        # 允许展示思考过程
        "Think step by step.",
        # 明确输出格式：最后一行给出答案
        "On the final separate line, output exactly: Answer: X (X is A‑J).",
    ])


def _image_index_from_path(p: str) -> int:
    name = Path(p).name
    m = re.search(r"image(?:_(\d+))?\.[a-zA-Z0-9]+$", name)
    if not m:
        return 1
    g = m.group(1)
    return int(g) if g and g.isdigit() else 1


@dataclass
class Example:
    uuid: str
    subject: str
    gold: str
    images: List[str]


def load_examples(jsonl_path: str, root_dir: Optional[str] = None, subject: Optional[str] = None) -> List[Example]:
    root = Path(root_dir or ".").resolve()
    df = pd.read_json(jsonl_path, lines=True)
    if df.empty:
        return []
    for col in ["uuid", "subject", "ori_ans", "image"]:
        if col not in df.columns:
            df[col] = None
    df["uuid"] = df["uuid"].astype(str)
    if subject:
        df = df[df["subject"].astype(str) == subject]
    by_uuid: Dict[str, Example] = {}
    for _, r in df.iterrows():
        rid = str(r.get("uuid") or "")
        if not rid:
            continue
        path = str(r.get("image") or "")
        if not path:
            continue
        ap = str((root / path).resolve()) if not os.path.isabs(path) else path
        subj = str(r.get("subject") or "")
        gold = letter_from_ori_ans(str(r.get("ori_ans") or ""))
        ex = by_uuid.get(rid)
        if not ex:
            ex = Example(uuid=rid, subject=subj, gold=gold, images=[])
            by_uuid[rid] = ex
        ex.images.append(ap)
    # 按文件名顺序排序图片
    for ex in by_uuid.values():
        ex.images = sorted(ex.images, key=_image_index_from_path)
    return list(by_uuid.values())


def load_done_uuids(results_path: Path) -> set:
    """Load completed uuids from an existing results JSONL (resume support).

    - Ignores malformed lines.
    - Returns an empty set if file missing/empty.
    """
    done: set = set()
    try:
        if results_path.exists() and results_path.stat().st_size > 0:
            with results_path.open('r', encoding='utf-8') as fr:
                for ln in fr:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        u = obj.get('uuid')
                        if isinstance(u, str) and u:
                            done.add(u)
    except Exception:
        # Be permissive: if we cannot read, don't skip anything
        return set()
    return done


class RateLimiter:
    """Sliding-window RPM limiter (thread-safe)."""

    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.window = 60.0
        self.lock = threading.Lock()
        self.calls = deque()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and (now - self.calls[0]) >= self.window:
                    self.calls.popleft()
                if len(self.calls) < self.rpm:
                    self.calls.append(now)
                    return
                wait = self.window - (now - self.calls[0])
            time.sleep(max(0.01, min(wait, 1.0)))


def _dump_response_json(resp: Any) -> str:
    try:
        if hasattr(resp, "model_dump_json"):
            return resp.model_dump_json()
        if hasattr(resp, "model_dump"):
            return json.dumps(resp.model_dump(), ensure_ascii=False)
        return json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o)), ensure_ascii=False)
    except Exception:
        try:
            return str(resp)
        except Exception:
            return ""


def call_model(client, model: str, image_paths: List[str], reasoning_effort: str, limiter: RateLimiter, temperature: float) -> (str, str):
    """Return (output_text, full_response_json)."""
    content = [{"type": "input_text", "text": build_prompt_image_only()}]
    for p in image_paths[:10]:
        content.append({"type": "input_image", "image_url": encode_image_to_data_url(p)})
    # Obey RPM for the create call
    limiter.acquire()
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": reasoning_effort, "summary": "detailed"},
        # temperature=temperature,
    )
    out_text = getattr(resp, "output_text", None) or ""
    raw_json = _dump_response_json(resp)
    return out_text, raw_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="vision_images.jsonl 路径")
    ap.add_argument("--subject", default=None, help="仅评测指定学科（可选）")
    ap.add_argument("--model", default="gpt-5", help="OpenAI 模型名，例如 gpt-5 / gpt-5-mini / gpt-5-chat-latest")
    ap.add_argument("--reasoning-effort", default="medium", choices=["minimal", "low", "medium", "high"])  # GPT‑5
    ap.add_argument("--verbosity", default="low", choices=["minimal", "low", "medium", "high"])  # 若模型支持
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="仅评测前 N 条（0 为全部）")
    ap.add_argument("--out", default="results_gpt5_vision.jsonl")
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--rpm", type=int, default=60, help="每分钟最大 API 调用次数（所有 HTTP 调用均计入）")
    ap.add_argument("--workers", type=int, default=4, help="并发线程数")
    args = ap.parse_args()

    if OpenAI is None:
        raise SystemExit("openai 未安装：请先 `pip install -U openai`")
    client = OpenAI(base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    timeout=1200)
    if not client.api_key:
        raise SystemExit("未检测到 OPENAI_API_KEY 环境变量")

    examples = load_examples(args.jsonl, subject=args.subject)
    out_path = Path(args.out)
    # Resume: skip uuids already in results
    already_done = load_done_uuids(out_path)
    if already_done:
        print(f"[i] Resuming: found {len(already_done)} completed uuids in {out_path}")
    if already_done:
        examples = [ex for ex in examples if ex.uuid not in already_done]
        print(f"[i] Remaining after resume filter: {len(examples)}")
    if args.limit > 0 and len(examples) > args.limit:
        examples = examples[: args.limit]
    if not examples:
        print("[!] 没有可评测样本（检查 jsonl/subject 过滤）")
        return

    limiter = RateLimiter(args.rpm)
    workers = int(args.workers) if args.workers > 0 else max(1, min(8, args.rpm))
    total = 0
    correct = 0
    completed = 0
    lock = threading.Lock()

    def _task(ex: Example):
        imgs = ex.images[: args.max_images]
        try:
            text, raw = call_model(
                client,
                model=args.model,
                image_paths=imgs,
                reasoning_effort=args.reasoning_effort,
                limiter=limiter,
                temperature=args.temperature,
            )
            pred = parse_answer_letter(text)
            err = ""
        except Exception as e:
            text, raw, pred = (f"<error> {type(e).__name__}: {e}", "", "")
            err = f"{type(e).__name__}: {e}"
        gold = ex.gold
        ok = bool(pred and gold and pred == gold)
        rec = {
            "uuid": ex.uuid,
            "subject": ex.subject,
            "gold": gold,
            "pred": pred,
            "ok": ok,
            "n_images": len(imgs),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "response_text": text,
            "response": raw,
            "error": err,
        }
        return rec, ok

    # Append mode + per-line flush+fsync for durability and crash safety
    with out_path.open("a", encoding="utf-8") as fw, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_task, ex) for ex in examples]
        for i, fut in enumerate(as_completed(futs), 1):
            rec, ok = fut.result()
            with lock:
                total += 1
                correct += int(ok)
                completed += 1
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fw.flush()
                try:
                    os.fsync(fw.fileno())
                except Exception:
                    pass
                if completed % 20 == 0 or completed == len(futs):
                    acc = 100.0 * correct / max(1, total)
                    print(f"[..] {completed}/{len(futs)}  acc={acc:.2f}%  (correct={correct})")

    final_acc = 100.0 * correct / max(1, total)
    print(f"[✓] Done. Acc={final_acc:.2f}%  wrote: {out_path.resolve()}  (RPM={args.rpm}, workers={workers})")


if __name__ == "__main__":
    main()
