# ruff: noqa  -- throwaway spike (smart-context spec step 0), not product code
"""Throwaway spike (smart-context spec §10 step 0): load embeddinggemma ONNX, time it, compare variants.

uv run python scripts/embed_spike.py
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from huggingface_hub import list_repo_files, snapshot_download
from tokenizers import Tokenizer

load_dotenv()
REPO = os.environ.get("EMBED_MODEL", "onnx-community/embeddinggemma-300m-ONNX")
TOKEN = os.environ.get("HF_TOKEN") or None
DOC = "title: none | text: {}"
QUERY = "task: search result | query: {}"
PAIRS = [
    ("Tôi có con chó tên là Bin.", "I have a dog named Bin."),
    ("Tôi sống ở Sài Gòn, quận 3.", "I live in Saigon, in District 3."),
    ("Em gái tôi tên là Mai.", "My younger sister is called Mai."),
    ("Tôi thích cà phê sữa đá.", "My favourite drink is iced coffee with milk."),
    ("Tuần sau tôi thi toán.", "I have a math exam next week."),
]


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p))


def main() -> None:
    files = list_repo_files(REPO, token=TOKEN)
    onnx_files = [f for f in files if f.endswith(".onnx")]
    print("onnx files:", onnx_files)
    pick = [f for f in onnx_files if f in ("onnx/model.onnx", "onnx/model_quantized.onnx")]
    path = Path(
        snapshot_download(
            REPO,
            token=TOKEN,
            allow_patterns=["*.json", "*.model", *pick, *[p + "_data" for p in pick]],
            local_dir="models/embeddinggemma",  # real files: onnxruntime rejects HF cache symlinks
        )
    )
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    tok.enable_truncation(256)
    tok.enable_padding()

    texts = [
        json.loads(line).get("say")
        for line in Path("tests/replay/memory_eval.jsonl").read_text().splitlines()
        if line.strip()
    ]
    texts = [t for t in texts if t]
    out: dict[str, np.ndarray] = {}
    for f in pick:
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(path / f), providers=["CPUExecutionProvider"])
        load_s = time.perf_counter() - t0
        names_in = [i.name for i in sess.get_inputs()]
        names_out = [o.name for o in sess.get_outputs()]

        def embed(batch: list[str], tpl: str) -> np.ndarray:
            enc = tok.encode_batch([tpl.format(b) for b in batch])
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            res = sess.run(None, {k: v for k, v in feed.items() if k in names_in})
            vec = (
                res[names_out.index("sentence_embedding")]
                if "sentence_embedding" in names_out
                else res[-1]
            )
            vec = vec[:, :256]
            return vec / np.linalg.norm(vec, axis=1, keepdims=True)

        embed(["warm up"], QUERY)
        q_ms = []
        for t in texts * 3:
            t0 = time.perf_counter()
            embed([t], QUERY)
            q_ms.append(1000 * (time.perf_counter() - t0))
        b_ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            embed(texts[:16], DOC)
            b_ms.append(1000 * (time.perf_counter() - t0))
        out[f] = embed(texts, DOC)
        vn = embed([a for a, _ in PAIRS], DOC)
        en = embed([b for _, b in PAIRS], DOC)
        sims = vn @ en.T
        print(f"\n== {f}")
        print(" inputs", names_in, "outputs", names_out)
        print(
            f" load {load_s:.2f}s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB"
        )
        print(
            f" query p50/p95 {pct(q_ms, 50):.1f}/{pct(q_ms, 95):.1f} ms; 16-doc batch p50 {pct(b_ms, 50):.1f} ms"
        )
        print(" VN↔EN matching pairs", np.round(np.diag(sims), 3).tolist())
        print(" VN↔EN best non-matching", round(float((sims - np.eye(5) * 9).max()), 3))
        # push timing: 5000 vectors, mask 2500, top 3
        rng = np.random.default_rng(0)
        mat = rng.normal(size=(5000, 256)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        allowed = np.zeros(5000, bool)
        allowed[::2] = True
        p_ms = []
        for t in texts * 2:
            t0 = time.perf_counter()
            q = embed([t], QUERY)[0]
            s = mat @ q
            s[~allowed] = -2
            np.argpartition(-s, 3)[:3]
            p_ms.append(1000 * (time.perf_counter() - t0))
        print(
            f" push (embed+mask+matmul+top3, 5000 vecs) p50/p95 {pct(p_ms, 50):.1f}/{pct(p_ms, 95):.1f} ms"
        )
    if len(out) == 2:
        a, b = out.values()
        print("\nfp32 vs quantized min cosine", round(float((a * b).sum(1).min()), 4))
    sizes = {f: (path / f).stat().st_size / 1e6 for f in pick}
    data = {
        f: (path / (f + "_data")).stat().st_size / 1e6
        for f in pick
        if (path / (f + "_data")).exists()
    }
    print("sizes MB", sizes, "external data MB", data, "\npath", path)


if __name__ == "__main__":
    main()
