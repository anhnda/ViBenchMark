"""
Vietnamese ASR Benchmark Extractor
====================================
Tạo hai tập dữ liệu âm thanh tiếng Việt chuẩn từ BUD500:

  1. CALIBRATION SET (~128-512 samples)
     Dùng để calibrate quantization (AWQ, GPTQ, SmoothQuant...)
     Tương tự vai trò của wikitext-2 train split trong LLM quantization.
     Yêu cầu: đa dạng, không trùng eval set, nhỏ gọn.

  2. EVAL SET (~600 samples, ~100MB)
     Dùng để đánh giá WER/CER sau quantization.
     Tương tự vai trò của wikitext-2 test split.

Nguồn   : linhtran92/viet_bud500 (train + test split, Parquet-native)
Strategy: Random sampling cố định seed, không chồng lấp giữa 2 tập.

Output structure:
  vi_asr_bench/
    calib/
      metadata.json          # thông tin tái hiện
      calib_N128_seed42.pkl  # list[dict] calibration samples
      wavs/                  # audio files (16kHz mono WAV)
        sample_0000.wav
        ...
    eval/
      metadata.json
      eval_N600_seed42.pkl   # list[dict] eval samples
      wavs/
        sample_0000.wav
        ...

Format mỗi sample trong pkl:
  {
    "id":           "sample_0000",
    "wav_path":     "vi_asr_bench/calib/wavs/sample_0000.wav",
    "transcript":   "xin chào thế giới",
    "duration_s":   3.21,
    "sampling_rate": 16000,
    "split":        "train",   # nguồn từ BUD500 split nào
  }

Usage:
  pip install datasets soundfile tqdm numpy
  python extract_vi_asr_benchmark.py
  python extract_vi_asr_benchmark.py --n-calib 256 --n-eval 600 --output-dir ./vi_asr_bench
"""

import io
import json
import pickle
import random
import argparse
import unicodedata
from pathlib import Path

import numpy as np
from tqdm import tqdm

# soundfile để ghi WAV
try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

# ---------------------------------------------------------------------------
# Config mặc định — tương tự LLM quantization convention
# ---------------------------------------------------------------------------
N_CALIB_DEFAULT = 128    # chuẩn AWQ/GPTQ: 128 samples
N_EVAL_DEFAULT  = 600    # ~100MB audio
SEED            = 42
SAMPLE_RATE     = 16_000
MIN_DUR_S       = 1.0    # lọc audio quá ngắn
MAX_DUR_S       = 20.0   # lọc audio quá dài


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def array_to_wav_bytes(array: "np.ndarray", sr: int = SAMPLE_RATE) -> bytes:
    """Chuyển numpy array sang WAV bytes (in-memory)."""
    import struct, wave
    # Chuẩn hoá về int16
    if array.dtype != np.int16:
        arr = np.clip(array, -1.0, 1.0)
        arr = (arr * 32767).astype(np.int16)
    else:
        arr = array

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


def save_wav(array: "np.ndarray", path: Path, sr: int = SAMPLE_RATE):
    """Lưu numpy array thành file WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_SF:
        sf.write(str(path), array, sr, subtype="PCM_16")
    else:
        wav_bytes = array_to_wav_bytes(array, sr)
        path.write_bytes(wav_bytes)


def resample_if_needed(array: "np.ndarray", src_sr: int, tgt_sr: int = SAMPLE_RATE) -> "np.ndarray":
    """Resample audio nếu sampling rate khác target."""
    if src_sr == tgt_sr:
        return array
    try:
        import librosa
        return librosa.resample(array.astype(np.float32), orig_sr=src_sr, target_sr=tgt_sr)
    except ImportError:
        # Fallback: linear interpolation đơn giản
        ratio = tgt_sr / src_sr
        n_out = int(len(array) * ratio)
        indices = np.linspace(0, len(array) - 1, n_out)
        return np.interp(indices, np.arange(len(array)), array.astype(np.float32))


# ---------------------------------------------------------------------------
# Text normalisation (dùng cho transcript lưu vào metadata)
# ---------------------------------------------------------------------------

def normalise_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.strip()


# ---------------------------------------------------------------------------
# BUD500 streaming collector
# ---------------------------------------------------------------------------

def collect_samples_from_split(
    hf_split: str,
    n_target: int,
    seed: int,
    exclude_ids: set = None,
    buffer_multiplier: int = 6,
) -> list[dict]:
    """
    Stream BUD500, lọc và thu thập n_target samples.

    Args:
        hf_split          : "train" hoặc "test"
        n_target          : số samples cần thu thập
        seed              : random seed
        exclude_ids       : set transcript đã dùng (tránh trùng giữa calib/eval)
        buffer_multiplier : thu thập buffer lớn hơn rồi sample
    """
    from datasets import load_dataset

    if exclude_ids is None:
        exclude_ids = set()

    buffer_size = min(n_target * buffer_multiplier, 8000)

    print(f"\n   📡 Streaming BUD500 [{hf_split}] — buffer {buffer_size} samples...")
    ds = load_dataset(
        "linhtran92/viet_bud500",
        split=hf_split,
        streaming=True,
    )

    buffer = []
    pbar = tqdm(ds, total=buffer_size, desc=f"   Collecting [{hf_split}]", unit="utt")

    for item in pbar:
        audio       = item["audio"]
        transcript  = item.get("transcription", "").strip()
        src_sr      = audio["sampling_rate"]
        arr         = audio["array"]
        duration    = len(arr) / src_sr

        # Lọc chất lượng
        if not transcript:
            continue
        if not (MIN_DUR_S <= duration <= MAX_DUR_S):
            continue
        if transcript in exclude_ids:   # tránh trùng với tập kia
            continue

        # Resample nếu cần
        if src_sr != SAMPLE_RATE:
            arr = resample_if_needed(arr, src_sr, SAMPLE_RATE)

        buffer.append({
            "audio_array":   arr.astype(np.float32),
            "sampling_rate": SAMPLE_RATE,
            "transcript":    normalise_vi(transcript),
            "duration_s":    duration,
            "source_split":  hf_split,
        })

        pbar.set_postfix({"buffered": len(buffer)})
        if len(buffer) >= buffer_size:
            break

    print(f"   Buffer: {len(buffer)} samples")

    # Random sample
    rng = random.Random(seed)
    selected = rng.sample(buffer, min(n_target, len(buffer)))
    return selected


# ---------------------------------------------------------------------------
# Saver — ghi pkl + WAV files + metadata
# ---------------------------------------------------------------------------

def save_split(
    samples: list[dict],
    out_dir: Path,
    split_name: str,         # "calib" hoặc "eval"
    n_samples: int,
    seed: int,
    save_wavs: bool = True,
) -> list[dict]:
    """
    Lưu samples ra local:
      - WAV files (optional, ~100MB cho eval)
      - PKL file (array in-memory, tiện dùng cho code)
      - metadata.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"

    records = []
    total_bytes = 0

    pbar = tqdm(enumerate(samples), total=len(samples),
                desc=f"   Saving [{split_name}]", unit="file")

    for i, s in pbar:
        sample_id = f"sample_{i:04d}"
        wav_path  = wav_dir / f"{sample_id}.wav"

        # Lưu WAV
        if save_wavs:
            save_wav(s["audio_array"], wav_path, SAMPLE_RATE)
            wav_bytes = wav_path.stat().st_size
            total_bytes += wav_bytes
            wav_path_str = str(wav_path)
        else:
            wav_path_str = ""

        record = {
            "id":            sample_id,
            "wav_path":      wav_path_str,
            "transcript":    s["transcript"],
            "duration_s":    round(s["duration_s"], 3),
            "sampling_rate": SAMPLE_RATE,
            "source_split":  s["source_split"],
        }
        records.append(record)
        pbar.set_postfix({"MB": f"{total_bytes/1e6:.1f}"})

    # PKL — lưu cả audio array để load nhanh không cần đọc file
    pkl_name = f"{split_name}_N{n_samples}_seed{seed}.pkl"
    pkl_path = out_dir / pkl_name

    pkl_data = []
    for rec, s in zip(records, samples):
        pkl_data.append({**rec, "audio_array": s["audio_array"]})

    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)

    # metadata.json
    total_dur = sum(s["duration_s"] for s in samples)
    meta = {
        "split":         split_name,
        "n_samples":     len(samples),
        "seed":          seed,
        "source":        "linhtran92/viet_bud500",
        "sampling_rate": SAMPLE_RATE,
        "total_duration_s":  round(total_dur, 1),
        "total_duration_min": round(total_dur / 60, 2),
        "total_size_mb":     round(total_bytes / 1e6, 1),
        "pkl_file":      pkl_name,
        "wav_dir":       "wavs/" if save_wavs else "",
        "min_dur_s":     MIN_DUR_S,
        "max_dur_s":     MAX_DUR_S,
        "purpose": (
            "ASR quantization calibration — feed audio arrays to model encoder"
            if split_name == "calib" else
            "ASR quantization evaluation — compute WER/CER vs transcript"
        ),
        "usage": (
            "pkl = pickle.load(open(pkl_file,'rb'))\n"
            "arrays = [s['audio_array'] for s in pkl]  # list of np.float32\n"
            "transcripts = [s['transcript'] for s in pkl]"
        ),
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n   ✅ [{split_name}] saved:")
    print(f"      {len(samples)} samples | {total_dur/60:.1f} min | {total_bytes/1e6:.1f} MB")
    print(f"      PKL : {pkl_path}")
    if save_wavs:
        print(f"      WAVs: {wav_dir}/")

    return records


# ---------------------------------------------------------------------------
# Loader helpers — dùng trong evaluator / quantizer
# ---------------------------------------------------------------------------

def load_calib_set(calib_dir: str = "./vi_asr_bench/calib",
                   n_calib: int = N_CALIB_DEFAULT,
                   seed: int = SEED) -> list[dict]:
    """
    Load calibration set từ local pkl.
    Trả về list[dict] với key 'audio_array' và 'transcript'.

    Dùng trong quantization script:
        samples = load_calib_set()
        arrays  = [torch.tensor(s['audio_array']).unsqueeze(0) for s in samples]
    """
    pkl_path = Path(calib_dir) / f"calib_N{n_calib}_seed{seed}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {pkl_path}.\n"
            f"Chạy: python extract_vi_asr_benchmark.py --n-calib {n_calib}"
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_eval_set(eval_dir: str = "./vi_asr_bench/eval",
                  n_eval: int = N_EVAL_DEFAULT,
                  seed: int = SEED) -> list[dict]:
    """
    Load eval set từ local pkl.
    Trả về list[dict] với key 'audio_array', 'transcript', 'wav_path'.

    Dùng trong vi_asr_eval.py:
        samples = load_eval_set()
        # thay thế cho load_bud500_sample()
    """
    pkl_path = Path(eval_dir) / f"eval_N{n_eval}_seed{seed}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {pkl_path}.\n"
            f"Chạy: python extract_vi_asr_benchmark.py --n-eval {n_eval}"
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_benchmark(
    n_calib:    int  = N_CALIB_DEFAULT,
    n_eval:     int  = N_EVAL_DEFAULT,
    seed:       int  = SEED,
    output_dir: str  = "./vi_asr_bench",
    save_wavs:  bool = True,
):
    out = Path(output_dir)

    print("=" * 65)
    print("🇻🇳  Vietnamese ASR Benchmark Extractor")
    print("=" * 65)
    print(f"Calibration : {n_calib} samples  (từ BUD500 train)")
    print(f"Eval        : {n_eval} samples  (từ BUD500 test)")
    print(f"Seed        : {seed}")
    print(f"Output      : {out.resolve()}")
    print(f"Save WAVs   : {save_wavs}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. EVAL SET — lấy từ BUD500 test split trước
    #    (test split nhỏ hơn, ít sample hơn → lấy trước để biết exclude)
    # ------------------------------------------------------------------
    print("\n[1/2] Building EVAL SET...")
    eval_samples = collect_samples_from_split(
        hf_split="test",
        n_target=n_eval,
        seed=seed,
    )

    # Tập transcript đã dùng — để calib không bị trùng
    eval_transcripts = set(s["transcript"] for s in eval_samples)

    eval_records = save_split(
        samples=eval_samples,
        out_dir=out / "eval",
        split_name="eval",
        n_samples=n_eval,
        seed=seed,
        save_wavs=save_wavs,
    )

    # ------------------------------------------------------------------
    # 2. CALIB SET — lấy từ BUD500 train split
    #    Tránh trùng transcript với eval set
    # ------------------------------------------------------------------
    print("\n[2/2] Building CALIBRATION SET...")
    calib_samples = collect_samples_from_split(
        hf_split="train",
        n_target=n_calib,
        seed=seed + 1,           # seed khác để shuffle khác
        exclude_ids=eval_transcripts,
    )

    save_split(
        samples=calib_samples,
        out_dir=out / "calib",
        split_name="calib",
        n_samples=n_calib,
        seed=seed,
        save_wavs=save_wavs,
    )

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("📊 SUMMARY")
    print("=" * 65)

    # So sánh với LLM convention
    print(f"\n{'Split':<14} {'N samples':>10} {'Duration':>12} {'~Size':>10}  {'Purpose'}")
    print("-" * 65)

    calib_dur = sum(s["duration_s"] for s in calib_samples)
    eval_dur  = sum(s["duration_s"] for s in eval_samples)
    calib_mb  = calib_dur * SAMPLE_RATE * 2 / 1e6   # 16-bit estimate
    eval_mb   = eval_dur  * SAMPLE_RATE * 2 / 1e6

    print(f"{'calib':<14} {len(calib_samples):>10} {calib_dur/60:>10.1f}m {calib_mb:>9.0f}MB  AWQ/GPTQ calibration")
    print(f"{'eval':<14} {len(eval_samples):>10}  {eval_dur/60:>10.1f}m  {eval_mb:>9.0f}MB  WER/CER evaluation")

    print(f"\n📁 Output: {out.resolve()}/")
    print(f"   calib/calib_N{n_calib}_seed{seed}.pkl")
    print(f"   eval/eval_N{n_eval}_seed{seed}.pkl")

    print("\n💡 Cách dùng:")
    print("   # Calibration (trong AWQ/GPTQ script)")
    print("   from extract_vi_asr_benchmark import load_calib_set")
    print("   calib = load_calib_set()  # list[dict]")
    print("   arrays = [s['audio_array'] for s in calib]")
    print()
    print("   # Evaluation (trong vi_asr_eval.py)")
    print("   from extract_vi_asr_benchmark import load_eval_set")
    print("   samples = load_eval_set()  # drop-in cho load_bud500_sample()")

    print("\n✅ Done!")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese ASR Benchmark Extractor — calib + eval from BUD500",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-calib",    type=int,  default=N_CALIB_DEFAULT,
                        help="Số samples calibration (128 hoặc 256 cho AWQ/GPTQ)")
    parser.add_argument("--n-eval",     type=int,  default=N_EVAL_DEFAULT,
                        help="Số samples eval (~100MB audio)")
    parser.add_argument("--seed",       type=int,  default=SEED)
    parser.add_argument("--output-dir", type=str,  default="./vi_asr_bench")
    parser.add_argument("--no-wavs",    action="store_true",
                        help="Không lưu WAV files (chỉ lưu pkl, tiết kiệm disk)")
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