"""
Vietnamese ASR Quantization Evaluator
======================================
Đánh giá WER/CER của model ASR tiếng Việt sau quantization.

Source  : linhtran92/viet_bud500 (test split, Parquet-native)
Sampling: Random ~600 utterances cố định seed (~100MB audio)
Metrics : WER (Word Error Rate) + CER (Character Error Rate)

Supported models:
  - Whisper (openai/whisper-*, distil-whisper/*)
  - wav2vec2 / HuBERT (CTC-based)
  - Bất kỳ model HuggingFace có .generate() hoặc pipeline("asr")

Usage:
  pip install datasets transformers jiwer tqdm numpy torch soundfile
  python vi_asr_eval.py --model openai/whisper-large-v3
  python vi_asr_eval.py --model openai/whisper-large-v3 --compare openai/whisper-medium
"""

import re
import json
import random
import pickle
import argparse
import unicodedata
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Sampling config
# ---------------------------------------------------------------------------
N_SAMPLES   = 600      # ~100MB audio ở 16kHz mono WAV (16-bit)
                       # 600 utterances × ~8s × 16000 × 2 bytes ≈ 154MB raw
                       # sau nén parquet thực tế ~80-120MB download
SEED        = 42
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Text normalisation — quan trọng cho WER/CER tiếng Việt
# ---------------------------------------------------------------------------

def normalise_vi(text: str) -> str:
    """
    Chuẩn hoá transcript tiếng Việt trước khi tính WER/CER.
    Áp dụng cho cả hypothesis (model output) lẫn reference (ground truth).

    Các bước:
      1. Unicode NFC — tránh lỗi dấu thanh tổ hợp vs tổ hợp sẵn
      2. Lowercase
      3. Bỏ dấu câu (giữ chữ cái, số, khoảng trắng)
      4. Chuẩn hoá khoảng trắng
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Giữ chữ cái Unicode (bao gồm tiếng Việt), số, khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Word Error Rate = (S + D + I) / N"""
    try:
        from jiwer import wer
        return wer(references, hypotheses)
    except ImportError:
        # Fallback thuần Python nếu không có jiwer
        return _wer_manual(references, hypotheses)


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """Character Error Rate — quan trọng hơn WER cho tiếng Việt"""
    try:
        from jiwer import cer
        return cer(references, hypotheses)
    except ImportError:
        return _cer_manual(references, hypotheses)


def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def _wer_manual(refs, hyps):
    total_words, total_errors = 0, 0
    for r, h in zip(refs, hyps):
        r_words = r.split()
        h_words = h.split()
        total_errors += _edit_distance(r_words, h_words)
        total_words  += max(len(r_words), 1)
    return total_errors / total_words if total_words else 0.0


def _cer_manual(refs, hyps):
    total_chars, total_errors = 0, 0
    for r, h in zip(refs, hyps):
        total_errors += _edit_distance(list(r), list(h))
        total_chars  += max(len(r), 1)
    return total_errors / total_chars if total_chars else 0.0


# ---------------------------------------------------------------------------
# Dataset: sample BUD500 test split
# ---------------------------------------------------------------------------

def load_bud500_sample(
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    cache_path: str = "./vi_asr_bench/bud500_sample.pkl",
) -> list[dict]:
    """
    Load và subsample BUD500 test split.
    Cache kết quả để tái sử dụng (tránh download lại).

    Mỗi sample: {"audio_array": np.ndarray, "sampling_rate": int, "transcript": str}
    """
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        print(f"📦 Loading từ cache: {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("📥 Downloading BUD500 test split (Parquet)...")
    print("   (Chỉ tải metadata trước, audio decode on-the-fly)")

    ds = load_dataset(
        "linhtran92/viet_bud500",
        split="test",
        streaming=True,   # stream để không tải hết 100GB
    )

    # Collect n_samples mẫu với seed cố định
    # Streaming không hỗ trợ shuffle seed trực tiếp trên HF →
    # thu thập buffer lớn hơn rồi random.sample
    BUFFER_SIZE = min(n_samples * 5, 5000)   # buffer 5× để sample đa dạng

    print(f"   Buffering {BUFFER_SIZE} samples...")
    buffer = []
    for item in tqdm(ds, total=BUFFER_SIZE, desc="   Buffering"):
        # Lọc utterance quá ngắn (<1s) hoặc quá dài (>20s)
        audio = item["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if 1.0 <= duration <= 20.0 and item.get("transcription", "").strip():
            buffer.append({
                "audio_array":   audio["array"],
                "sampling_rate": audio["sampling_rate"],
                "transcript":    item["transcription"].strip(),
                "duration":      duration,
            })
        if len(buffer) >= BUFFER_SIZE:
            break

    print(f"   Buffer: {len(buffer)} samples sau khi lọc")

    # Random sample cố định
    rng = random.Random(seed)
    samples = rng.sample(buffer, min(n_samples, len(buffer)))

    total_duration = sum(s["duration"] for s in samples)
    total_mb_est   = sum(
        len(s["audio_array"]) * 2 / 1e6   # 16-bit = 2 bytes/sample
        for s in samples
    )

    print(f"   ✅ {len(samples)} samples | "
          f"{total_duration/60:.1f} phút | "
          f"~{total_mb_est:.0f}MB raw audio")

    # Cache
    with open(cache, "wb") as f:
        pickle.dump(samples, f)
    print(f"   💾 Cached: {cache}")

    return samples


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class WhisperEvaluator:
    """Wrapper cho Whisper và Distil-Whisper."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.float16):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        print(f"\n🔧 Loading Whisper: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
            generate_kwargs={"language": "vi", "task": "transcribe"},
        )
        print("   ✅ Model loaded")

    def transcribe_batch(self, samples: list[dict], batch_size: int = 8) -> list[str]:
        """Transcribe danh sách samples, trả về list transcript."""
        audio_inputs = [
            {"array": s["audio_array"], "sampling_rate": s["sampling_rate"]}
            for s in samples
        ]
        results = []
        for i in tqdm(range(0, len(audio_inputs), batch_size),
                      desc="   Transcribing", unit="batch"):
            batch = audio_inputs[i : i + batch_size]
            out = self.pipe(batch, batch_size=batch_size)
            results.extend([o["text"] for o in out])
        return results


class CTCEvaluator:
    """Wrapper cho wav2vec2 / HuBERT (CTC decoder)."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.float16):
        from transformers import AutoModelForCTC, AutoProcessor

        print(f"\n🔧 Loading CTC model: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCTC.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        self.device = device
        self.dtype  = dtype
        print("   ✅ Model loaded")

    @torch.no_grad()
    def transcribe_batch(self, samples: list[dict], batch_size: int = 8) -> list[str]:
        results = []
        for i in tqdm(range(0, len(samples), batch_size),
                      desc="   Transcribing", unit="batch"):
            batch = samples[i : i + batch_size]
            arrays = [s["audio_array"] for s in batch]

            inputs = self.processor(
                arrays,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            logits = self.model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            transcripts = self.processor.batch_decode(pred_ids)
            results.extend(transcripts)
        return results


def load_evaluator(model_path: str, device: str = "cuda"):
    """Auto-detect model type và load evaluator phù hợp."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    arch = cfg.architectures[0] if cfg.architectures else ""

    if "Whisper" in arch or "whisper" in model_path.lower():
        return WhisperEvaluator(model_path, device)
    elif "CTC" in arch or "wav2vec2" in model_path.lower() or "hubert" in model_path.lower():
        return CTCEvaluator(model_path, device)
    else:
        # Fallback: thử Whisper pipeline
        print(f"⚠️  Không nhận dạng được architecture '{arch}', thử Whisper pipeline...")
        return WhisperEvaluator(model_path, device)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    samples: list[dict],
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """Chạy đánh giá WER/CER cho một model."""

    evaluator = load_evaluator(model_path, device)

    print(f"\n📊 Evaluating {len(samples)} samples...")
    hypotheses_raw = evaluator.transcribe_batch(samples, batch_size=batch_size)

    # Normalise
    references  = [normalise_vi(s["transcript"]) for s in samples]
    hypotheses  = [normalise_vi(h) for h in hypotheses_raw]

    # Lọc cặp rỗng (tránh chia 0)
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    refs_clean = [p[0] for p in pairs]
    hyps_clean = [p[1] for p in pairs]

    wer_score = compute_wer(refs_clean, hyps_clean)
    cer_score = compute_cer(refs_clean, hyps_clean)

    # Thống kê bổ sung
    total_words = sum(len(r.split()) for r in refs_clean)
    total_chars = sum(len(r) for r in refs_clean)
    avg_duration = np.mean([s["duration"] for s in samples])

    result = {
        "model":          model_path,
        "wer":            wer_score,
        "cer":            cer_score,
        "n_samples":      len(pairs),
        "total_words":    total_words,
        "total_chars":    total_chars,
        "avg_duration_s": avg_duration,
    }

    print(f"\n  WER : {wer_score*100:.2f}%")
    print(f"  CER : {cer_score*100:.2f}%")
    print(f"  Samples: {len(pairs)} | Words: {total_words:,} | Chars: {total_chars:,}")

    # Dọn VRAM
    del evaluator
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[dict]):
    print("\n" + "=" * 65)
    print("VIETNAMESE ASR QUANTIZATION EVALUATION — BUD500 SUBSAMPLE")
    print("=" * 65)

    if len(results) == 1:
        r = results[0]
        print(f"\nModel  : {r['model']}")
        print(f"WER    : {r['wer']*100:.2f}%")
        print(f"CER    : {r['cer']*100:.2f}%")
        print(f"Samples: {r['n_samples']} utterances")
        return

    # So sánh 2 models
    r0, r1 = results[0], results[1]
    delta_wer = (r0["wer"] - r1["wer"]) / r1["wer"] * 100
    delta_cer = (r0["cer"] - r1["cer"]) / r1["cer"] * 100

    col = 22
    print(f"\n{'Metric':<12} {r0['model'][:col]:<{col}} {r1['model'][:col]:<{col}} {'Delta':>10}")
    print("-" * 65)
    print(f"{'WER':<12} {r0['wer']*100:>{col}.2f}% {r1['wer']*100:>{col}.2f}% {delta_wer:>+9.2f}%")
    print(f"{'CER':<12} {r0['cer']*100:>{col}.2f}% {r1['cer']*100:>{col}.2f}% {delta_cer:>+9.2f}%")
    print(f"{'Samples':<12} {r0['n_samples']:>{col},} {r1['n_samples']:>{col},}")
    print("-" * 65)

    wer_winner = r0["model"] if r0["wer"] < r1["wer"] else r1["model"]
    cer_winner = r0["model"] if r0["cer"] < r1["cer"] else r1["model"]

    print(f"\n🏆 WER winner : {wer_winner.split('/')[-1]}")
    print(f"🏆 CER winner : {cer_winner.split('/')[-1]}")

    print("\n💡 Lưu ý: CER quan trọng hơn WER cho tiếng Việt")
    print("         (từ ngắn, dấu thanh dễ nhầm sau quantization)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese ASR Quantization Evaluator — BUD500 subsample",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",    required=True,
                        help="HuggingFace model path/ID (primary model)")
    parser.add_argument("--compare",  default="",
                        help="Model thứ hai để so sánh (optional)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help="Số utterances để đánh giá (~100MB)")
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default="./vi_asr_bench",
                        help="Thư mục cache dataset")
    parser.add_argument("--output",    default="",
                        help="Lưu kết quả JSON (optional)")
    args = parser.parse_args()

    print("=" * 65)
    print("🇻🇳  Vietnamese ASR Quantization Evaluator")
    print("=" * 65)
    print(f"Device    : {args.device}")
    print(f"N samples : {args.n_samples} utterances")
    print(f"Seed      : {args.seed}")
    print(f"Model     : {args.model}")
    if args.compare:
        print(f"Compare   : {args.compare}")
    print("=" * 65)

    # 1. Load dataset
    cache_path = str(Path(args.cache_dir) / f"bud500_n{args.n_samples}_seed{args.seed}.pkl")
    samples = load_bud500_sample(
        n_samples=args.n_samples,
        seed=args.seed,
        cache_path=cache_path,
    )

    # 2. Evaluate
    all_results = []

    result1 = evaluate_model(args.model, samples, args.batch_size, args.device)
    all_results.append(result1)

    if args.compare:
        result2 = evaluate_model(args.compare, samples, args.batch_size, args.device)
        all_results.append(result2)

    # 3. Report
    print_report(all_results)

    # 4. Save JSON
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Kết quả lưu tại: {out_path}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()