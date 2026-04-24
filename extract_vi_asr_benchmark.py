"""
Vietnamese ASR Benchmark Extractor (v2 — no torchcodec)
=========================================================
Fix: Bypass hoàn toàn HuggingFace audio decoder (torchcodec).
     Load parquet trực tiếp bằng pandas, decode audio bằng soundfile.

Usage:
  pip install soundfile tqdm numpy pandas pyarrow huggingface_hub
  python extract_vi_asr_benchmark.py
  python extract_vi_asr_benchmark.py --n-calib 256 --n-eval 600
"""

import io
import json
import pickle
import random
import argparse
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CALIB_DEFAULT = 256
N_EVAL_DEFAULT  = 600
SEED            = 42
SAMPLE_RATE     = 16_000
MIN_DUR_S       = 1.0
MAX_DUR_S       = 20.0

HF_REPO = "linhtran92/viet_bud500"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def decode_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes → (float32 array, sample_rate) dùng soundfile."""
    buf = io.BytesIO(raw_bytes)
    arr, sr = sf.read(buf, dtype="float32", always_2d=False)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)   # stereo → mono
    return arr, sr


def resample(arr: np.ndarray, src_sr: int, tgt_sr: int = SAMPLE_RATE) -> np.ndarray:
    if src_sr == tgt_sr:
        return arr
    n_out = int(len(arr) * tgt_sr / src_sr)
    return np.interp(
        np.linspace(0, len(arr) - 1, n_out),
        np.arange(len(arr)),
        arr,
    ).astype(np.float32)


def save_wav(arr: np.ndarray, path: Path, sr: int = SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sr, subtype="PCM_16")


def normalise_vi(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


# ---------------------------------------------------------------------------
# Load parquet trực tiếp — không qua HF audio decode
# ---------------------------------------------------------------------------

def list_parquet_urls(split: str) -> list[str]:
    """Lấy danh sách URL parquet files từ HuggingFace Hub."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    pattern = f"datasets/{HF_REPO}/data/{split}-*.parquet"
    files = fs.glob(pattern)
    urls = [
        "https://huggingface.co/datasets/{}/resolve/main/{}".format(
            HF_REPO, f.split(f"{HF_REPO}/")[-1]
        )
        for f in sorted(files)
    ]
    print(f"   Found {len(urls)} parquet file(s) for [{split}]")
    return urls


def iter_parquet_rows(urls: list[str]):
    """
    Đọc từng parquet file qua HTTP bằng pandas.
    KHÔNG gọi HF datasets → không cần torchcodec.
    Audio column là dict {"bytes": b"...", "path": "..."}.
    """
    import pandas as pd

    for url in urls:
        fname = url.split("/")[-1]
        try:
            df = pd.read_parquet(url, storage_options={"anon": True})
        except Exception as e:
            print(f"   ⚠️  Skip {fname}: {e}")
            continue
        print(f"   📄 {fname}: {len(df)} rows")
        for _, row in df.iterrows():
            yield row.to_dict()


def collect_samples(
    split: str,
    n_target: int,
    seed: int,
    exclude_transcripts: set = None,
    buffer_multiplier: int = 5,
) -> list[dict]:
    """Thu thập n_target samples từ BUD500 split."""

    if exclude_transcripts is None:
        exclude_transcripts = set()

    buffer_size = min(n_target * buffer_multiplier, 6000)
    urls = list_parquet_urls(split)

    print(f"   Target buffer: {buffer_size} samples...")
    buffer = []
    pbar = tqdm(total=buffer_size, desc=f"   [{split}]", unit="utt")

    for row in iter_parquet_rows(urls):
        transcript = str(row.get("transcription", "")).strip()
        if not transcript or transcript in exclude_transcripts:
            continue

        # Audio field: dict với key "bytes"
        audio_field = row.get("audio", {})
        raw_bytes = (
            audio_field.get("bytes")
            if isinstance(audio_field, dict)
            else None
        )
        if not raw_bytes:
            continue

        try:
            arr, src_sr = decode_audio_bytes(raw_bytes)
        except Exception:
            continue

        duration = len(arr) / src_sr
        if not (MIN_DUR_S <= duration <= MAX_DUR_S):
            continue

        if src_sr != SAMPLE_RATE:
            arr = resample(arr, src_sr, SAMPLE_RATE)

        buffer.append({
            "audio_array":   arr,
            "sampling_rate": SAMPLE_RATE,
            "transcript":    normalise_vi(transcript),
            "duration_s":    round(duration, 3),
            "source_split":  split,
        })
        pbar.update(1)

        if len(buffer) >= buffer_size:
            break

    pbar.close()
    print(f"   Buffered {len(buffer)} valid samples")

    rng = random.Random(seed)
    selected = rng.sample(buffer, min(n_target, len(buffer)))
    return selected


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_split(
    samples: list[dict],
    out_dir: Path,
    split_name: str,
    n_samples: int,
    seed: int,
    save_wavs: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"
    records = []
    total_bytes = 0

    pbar = tqdm(enumerate(samples), total=len(samples),
                desc=f"   Saving [{split_name}]", unit="file")

    for i, s in pbar:
        sid      = f"sample_{i:04d}"
        wav_path = wav_dir / f"{sid}.wav"

        if save_wavs:
            save_wav(s["audio_array"], wav_path)
            total_bytes += wav_path.stat().st_size

        records.append({
            "id":            sid,
            "wav_path":      str(wav_path) if save_wavs else "",
            "transcript":    s["transcript"],
            "duration_s":    s["duration_s"],
            "sampling_rate": SAMPLE_RATE,
            "source_split":  s["source_split"],
        })

    # PKL với audio arrays
    pkl_name = f"{split_name}_N{n_samples}_seed{seed}.pkl"
    pkl_path = out_dir / pkl_name
    pkl_data = [
        {**rec, "audio_array": s["audio_array"]}
        for rec, s in zip(records, samples)
    ]
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)

    # metadata
    total_dur = sum(s["duration_s"] for s in samples)
    meta = {
        "split":              split_name,
        "n_samples":          len(samples),
        "seed":               seed,
        "source":             HF_REPO,
        "sampling_rate":      SAMPLE_RATE,
        "total_duration_s":   round(total_dur, 1),
        "total_duration_min": round(total_dur / 60, 2),
        "total_size_mb":      round(total_bytes / 1e6, 1),
        "pkl_file":           pkl_name,
        "purpose": (
            "calibration — feed audio_array to model encoder for AWQ/GPTQ"
            if split_name == "calib" else
            "evaluation — compute WER/CER against transcript"
        ),
        "usage": (
            "import pickle\n"
            "samples = pickle.load(open(pkl_file, 'rb'))\n"
            "arrays  = [s['audio_array'] for s in samples]   # np.float32\n"
            "texts   = [s['transcript']  for s in samples]"
        ),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"   ✅ [{split_name}] {len(samples)} samples | "
          f"{total_dur/60:.1f} min | {total_bytes/1e6:.1f} MB")
    print(f"      PKL  : {pkl_path}")
    if save_wavs:
        print(f"      WAVs : {wav_dir}/")


# ---------------------------------------------------------------------------
# Public loaders — dùng trong evaluator / quantizer
# ---------------------------------------------------------------------------

def load_calib_set(
    calib_dir: str = "./vi_asr_bench/calib",
    n_calib: int = N_CALIB_DEFAULT,
    seed: int = SEED,
) -> list[dict]:
    """
    Load calibration set từ local pkl.
    Dùng trong AWQ/GPTQ script:
        samples = load_calib_set()
        arrays  = [torch.tensor(s['audio_array']).unsqueeze(0) for s in samples]
    """
    pkl = Path(calib_dir) / f"calib_N{n_calib}_seed{seed}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {pkl}.\n"
            f"Chạy: python extract_vi_asr_benchmark.py --n-calib {n_calib}"
        )
    with open(pkl, "rb") as f:
        return pickle.load(f)


def load_eval_set(
    eval_dir: str = "./vi_asr_bench/eval",
    n_eval: int = N_EVAL_DEFAULT,
    seed: int = SEED,
) -> list[dict]:
    """
    Load eval set từ local pkl.
    Drop-in thay thế load_bud500_sample() trong vi_asr_eval.py:
        samples = load_eval_set()
    """
    pkl = Path(eval_dir) / f"eval_N{n_eval}_seed{seed}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {pkl}.\n"
            f"Chạy: python extract_vi_asr_benchmark.py --n-eval {n_eval}"
        )
    with open(pkl, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_benchmark(
    n_calib: int = N_CALIB_DEFAULT,
    n_eval:  int = N_EVAL_DEFAULT,
    seed:    int = SEED,
    output_dir: str = "./vi_asr_bench",
    save_wavs:  bool = True,
):
    out = Path(output_dir)

    print("=" * 65)
    print("🇻🇳  Vietnamese ASR Benchmark Extractor  (v2 — no torchcodec)")
    print("=" * 65)
    print(f"Calibration : {n_calib} samples  (BUD500 train)")
    print(f"Eval        : {n_eval} samples  (BUD500 test)")
    print(f"Seed        : {seed}")
    print(f"Output      : {out.resolve()}")
    print(f"Save WAVs   : {save_wavs}")
    print("=" * 65)

    # 1. Eval từ test split
    print("\n[1/2] Building EVAL SET (BUD500 test)...")
    eval_samples = collect_samples("test", n_eval, seed)
    eval_transcripts = {s["transcript"] for s in eval_samples}
    save_split(eval_samples, out / "eval", "eval", n_eval, seed, save_wavs)

    # 2. Calib từ train split, tránh trùng eval
    print("\n[2/2] Building CALIBRATION SET (BUD500 train)...")
    calib_samples = collect_samples(
        "train", n_calib, seed + 1,
        exclude_transcripts=eval_transcripts,
    )
    save_split(calib_samples, out / "calib", "calib", n_calib, seed, save_wavs)

    # Summary
    calib_dur = sum(s["duration_s"] for s in calib_samples)
    eval_dur  = sum(s["duration_s"] for s in eval_samples)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  calib : {len(calib_samples):>4} samples | {calib_dur/60:>5.1f} min → AWQ/GPTQ calibration")
    print(f"  eval  : {len(eval_samples):>4} samples | {eval_dur/60:>5.1f} min → WER/CER evaluation")
    print(f"\n  calib pkl : {out}/calib/calib_N{n_calib}_seed{seed}.pkl")
    print(f"  eval  pkl : {out}/eval/eval_N{n_eval}_seed{seed}.pkl")
    print("\n✅ Done!")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese ASR Benchmark Extractor (no torchcodec)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-calib",    type=int,  default=N_CALIB_DEFAULT,
                        help="Số samples calibration (128 hoặc 256 cho AWQ/GPTQ)")
    parser.add_argument("--n-eval",     type=int,  default=N_EVAL_DEFAULT,
                        help="Số samples eval (~100MB audio)")
    parser.add_argument("--seed",       type=int,  default=SEED)
    parser.add_argument("--output-dir", type=str,  default="./vi_asr_bench")
    parser.add_argument("--no-wavs",    action="store_true",
                        help="Không lưu WAV files, chỉ lưu pkl (tiết kiệm disk)")
    args = parser.parse_args()

    build_benchmark(
        n_calib=args.n_calib,
        n_eval=args.n_eval,
        seed=args.seed,
        output_dir=args.output_dir,
        save_wavs=not args.no_wavs,
    )


if __name__ == "__main__":
    main()