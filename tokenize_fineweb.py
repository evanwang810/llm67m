#!/usr/bin/env python
"""Pre-tokenize FineWeb-Edu into uint16 .bin shards, locally, on CPU.

    python tokenize_fineweb.py --out-dir data/fineweb-edu --max-tokens 4e9

Notes for your machine:
  * Tokenization is pure CPU. The Intel Arc iGPU cannot help with BPE, so do not
    bother installing anything for it. tiktoken is already SIMD-fast C.
  * The real bottleneck is your download speed. 4B GPT-2 tokens is roughly
    11 GB of text and about 8 GB of output .bin.
  * Safe to kill and rerun. It records how many documents it consumed and
    resumes by skipping them, and each finished shard is written atomically.
  * Upload the whole out-dir as one Kaggle Dataset. Keep individual shards
    around 1 to 2 GB so the browser uploader stays happy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

_enc = None


def _worker_init(encoding: str) -> None:
    global _enc
    import tiktoken

    _enc = tiktoken.get_encoding(encoding)


def _encode(text: str) -> np.ndarray:
    ids = _enc.encode_ordinary(text)
    ids.append(_enc.eot_token)  # document separator
    arr = np.asarray(ids, dtype=np.uint32)
    if arr.max(initial=0) >= 2**16:
        raise ValueError("token id does not fit in uint16")
    return arr.astype(np.uint16)


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.n = 0
        self.names: list[str] = []
        self.total = 0

    def add(self, arr: np.ndarray) -> None:
        pos = 0
        while pos < len(arr):
            room = self.shard_tokens - self.n
            take = min(room, len(arr) - pos)
            self.buf[self.n : self.n + take] = arr[pos : pos + take]
            self.n += take
            pos += take
            self.total += take
            if self.n == self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if self.n == 0:
            return
        name = f"{self.prefix}_{len(self.names):03d}.bin"
        tmp = self.out_dir / (name + ".tmp")
        self.buf[: self.n].tofile(tmp)
        os.replace(tmp, self.out_dir / name)
        print(f"  wrote {name}  {self.n:,} tokens  {(self.n * 2) / 1e9:.2f} GB")
        self.names.append(name)
        self.n = 0


def write_meta(out_dir: Path, encoding: str, vocab_size: int, train: list[str], val: list[str],
               docs_consumed: int, tokens: int) -> None:
    meta = {
        "tokenizer": encoding,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "shards": {"train": train, "val": val},
        "docs_consumed": docs_consumed,
        "total_tokens": tokens,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/fineweb-edu")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-key", default="text")
    p.add_argument("--encoding", default="gpt2")
    p.add_argument("--max-tokens", type=float, default=4e9)
    p.add_argument("--val-tokens", type=float, default=5e6)
    p.add_argument("--shard-tokens", type=float, default=5e8, help="1 GB per 5e8 tokens")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--batch", type=int, default=256, help="documents per worker chunk")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_tokens, val_tokens, shard_tokens = int(args.max_tokens), int(args.val_tokens), int(args.shard_tokens)

    skip_docs, existing_train, existing_val, done_tokens = 0, [], [], 0
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        skip_docs = old.get("docs_consumed", 0)
        existing_train = old["shards"]["train"]
        existing_val = old["shards"]["val"]
        done_tokens = old.get("total_tokens", 0)
        print(f"resuming: {done_tokens / 1e9:.3f}B tokens already written, "
              f"skipping {skip_docs:,} documents")
        # Check before importing datasets, so a re-run with complete shards
        # needs neither the package nor the network.
        if done_tokens >= max_tokens:
            print("already at the requested token count, nothing to do")
            return

    import tiktoken
    from datasets import load_dataset

    vocab_size = tiktoken.get_encoding(args.encoding).n_vocab

    print(f"streaming {args.dataset} [{args.name}] with {args.workers} workers")
    ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)
    if skip_docs:
        ds = ds.skip(skip_docs)

    val_writer = ShardWriter(out_dir, "val", shard_tokens)
    val_writer.names = list(existing_val)
    train_writer = ShardWriter(out_dir, "train", shard_tokens)
    train_writer.names = list(existing_train)

    need_val = val_tokens if not existing_val else 0
    docs = skip_docs
    tokens = done_tokens
    t0 = time.time()

    def texts():
        for row in ds:
            yield row[args.text_key]

    with mp.Pool(args.workers, initializer=_worker_init, initargs=(args.encoding,)) as pool:
        try:
            for arr in pool.imap(_encode, texts(), chunksize=args.batch):
                docs += 1
                if need_val > 0:
                    val_writer.add(arr)
                    need_val -= len(arr)
                    if need_val <= 0:
                        val_writer.flush()
                        print(f"  val split done: {val_writer.total:,} tokens")
                    continue
                train_writer.add(arr)
                tokens += len(arr)
                if docs % 20000 == 0:
                    el = time.time() - t0
                    rate = (tokens - done_tokens) / max(1e-6, el)
                    eta = (max_tokens - tokens) / max(1.0, rate)
                    print(f"{tokens / 1e9:6.3f}B / {max_tokens / 1e9:.1f}B tokens | "
                          f"{docs:,} docs | {rate / 1e6:.2f}M tok/s | eta {eta / 3600:.1f}h",
                          flush=True)
                    write_meta(out_dir, args.encoding, vocab_size, train_writer.names,
                               val_writer.names, docs, tokens)
                if tokens >= max_tokens:
                    break
        except KeyboardInterrupt:
            print("\ninterrupted, flushing what we have")

    train_writer.flush()
    val_writer.flush()
    write_meta(out_dir, args.encoding, vocab_size, train_writer.names, val_writer.names, docs, tokens)
    print(f"\ndone: {tokens / 1e9:.3f}B train tokens in {len(train_writer.names)} shards, "
          f"{val_writer.total:,} val tokens")
    print(f"total on disk: {sum(f.stat().st_size for f in out_dir.glob('*.bin')) / 1e9:.2f} GB")
    print(f"upload {out_dir} as a Kaggle Dataset next")


if __name__ == "__main__":
    mp.freeze_support()
    main()
