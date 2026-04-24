"""
Vietnamese Quantization Evaluation Dataset Builder
====================================================
Tạo bộ dữ liệu tiếng Việt chuẩn để đánh giá perplexity cho quantization LLM.

Nguồn dữ liệu:
  - Primary  : wikipedia/20231101.vi  (toàn bộ Wikipedia tiếng Việt)
  - Fallback  : cc100 (vi), oscar (vi), MC4 (vi)

Kích thước mục tiêu (tương đương WikiText-2 tiếng Anh):
  - Test  : ~2.08M ký tự  (~280k token BPE)   — tương đương wt2 test
  - Valid : ~218k  ký tự  (~29k  token BPE)   — tương đương wt2 valid
  - Train : ~2.08M ký tự  (~280k token BPE)   — tương đương wt2 train

Methodology:
  - Xử lý text theo chuẩn WikiText (giữ văn xuôi liên tục, bỏ table/template)
  - Ghép các đoạn văn thành stream liên tục (giống cách wt2 được xây dựng)
  - Lưu dưới dạng plain-text (.txt) và pickle (.pkl) tương thích với validator
  - Ghi metadata JSON để tái hiện

Usage:
  pip install datasets tqdm numpy
  python build_vi_benchmark.py [--output-dir ./vi_bench] [--seed 42]
"""

import os
import re
import json
import pickle
import random
import argparse
import unicodedata
from pathlib import Path

import numpy as np
from tqdm import tqdm
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Target sizes (chars) — calibrated against WikiText-2 English splits
# ---------------------------------------------------------------------------
TARGETS = {
    "test":  2_080_000,
    "valid":   218_000,
    "train": 2_080_000,
}

TOTAL_TARGET = sum(TARGETS.values())   # ~4.38M chars


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_WIKI_TEMPLATE = re.compile(r"\{\{[^}]*\}\}")         # {{template}}
_WIKI_TABLE    = re.compile(r"\{\|.*?\|\}", re.DOTALL) # {| table |}
_WIKI_LINK     = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")  # [[link|text]] -> text
_WIKI_REF      = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
_HTML_TAG      = re.compile(r"<[^>]+>")
_EQUALS_LINE   = re.compile(r"^=+\s.*\s=+$", re.MULTILINE)  # section headers
_PUNCT_REPEAT  = re.compile(r"([.!?,;:]){3,}")
_WHITESPACE    = re.compile(r"[ \t]+")


def clean_wiki_text(text: str) -> str:
    """
    Làm sạch văn bản Wikipedia tiếng Việt, giữ lại văn xuôi chất lượng cao.
    Chuẩn hoá tương tự pipeline của WikiText-103.
    """
    # Loại bỏ template, table, ref, HTML
    text = _WIKI_TEMPLATE.sub("", text)
    text = _WIKI_TABLE.sub("", text)
    text = _WIKI_REF.sub("", text)
    text = _HTML_TAG.sub("", text)

    # Giữ nội dung link, bỏ markup
    text = _WIKI_LINK.sub(r"\1", text)

    # Chuẩn hoá Unicode (NFC) — quan trọng cho tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # Loại bỏ dòng header === ... ===  (giữ nội dung, bỏ dấu =)
    text = _EQUALS_LINE.sub("", text)

    # Dọn whitespace
    text = _WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


def is_quality_paragraph(para: str, min_chars: int = 150) -> bool:
    """
    Lọc đoạn văn chất lượng thấp:
      - Quá ngắn
      - Quá nhiều ký tự đặc biệt (>25%) → có thể là code/template sót
      - Quá ít chữ cái tiếng Việt
    """
    if len(para) < min_chars:
        return False

    # Đếm ký tự chữ cái
    alpha = sum(1 for c in para if c.isalpha())
    if alpha / len(para) < 0.55:
        return False

    # Kiểm tra ít nhất một nguyên âm tiếng Việt đặc trưng
    vi_vowels = set("àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
    has_vi = any(c in vi_vowels for c in para.lower())
    if not has_vi:
        return False

    return True


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

"""
PATCH: Thay thế hai hàm loader trong build_vi_benchmark.py
Lý do: datasets >= 2.20 không còn hỗ trợ loading script (wikipedia.py, cc100.py).
       Phải dùng các repo Parquet-native trên Hugging Face.

Nguồn thay thế:
  Primary  : wikimedia/wikipedia  20231101.vi   (Parquet-native, ~1.5GB)
  Fallback1: uonlp/CulturaX       vi            (Parquet-native, ~8GB stream)
  Fallback2: mc4                  vi            (Parquet-native, stream)
"""

# =====================================================================
# DÁN ĐÈ hai hàm này vào build_vi_benchmark.py
# =====================================================================

def collect_from_wikipedia_vi(target_chars: int, seed: int = 42) -> list[str]:
    """
    Thu thập văn bản từ Wikipedia tiếng Việt.
    Dùng 'wikimedia/wikipedia' — repo Parquet-native chính thức,
    thay thế cho 'wikipedia' (dùng loading script, không còn hỗ trợ).
    """
    print("\n📚 Loading Wikipedia tiếng Việt (wikimedia/wikipedia 20231101.vi)...")
    print("   (Lần đầu tải ~1.5GB, sau đó cache tự động)")

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.vi",
            split="train",
            # KHÔNG dùng trust_remote_code — đây là Parquet-native
        )
    except Exception as e:
        print(f"   ⚠️  Lỗi tải wikimedia/wikipedia: {e}")
        return []

    import random
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    from tqdm import tqdm
    paragraphs = []
    collected_chars = 0

    pbar = tqdm(indices, desc="   Xử lý bài viết", unit="art")
    for idx in pbar:
        article = ds[idx]
        text = clean_wiki_text(article.get("text", ""))

        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paras:
            if is_quality_paragraph(para):
                paragraphs.append(para)
                collected_chars += len(para)

        pbar.set_postfix({"chars": f"{collected_chars/1e6:.2f}M", "paras": len(paragraphs)})

        if collected_chars >= target_chars * 1.2:
            break

    print(f"   ✅ {len(paragraphs):,} đoạn văn ({collected_chars/1e6:.2f}M ký tự)")
    return paragraphs


def collect_from_cc100_vi(target_chars: int, seed: int = 42) -> list[str]:
    """
    Fallback 1: CulturaX tiếng Việt (uonlp/CulturaX) — Parquet-native.
    Thay thế cho 'cc100' (dùng loading script, không còn hỗ trợ).

    Nếu CulturaX cũng thất bại, thử tiếp mc4 tiếng Việt.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    # --- Thử CulturaX trước ---
    print("\n📚 Fallback 1: Loading CulturaX tiếng Việt (uonlp/CulturaX vi)...")
    try:
        ds = load_dataset(
            "uonlp/CulturaX",
            "vi",
            split="train",
            streaming=True,
        )
        paragraphs = []
        collected_chars = 0
        for item in tqdm(ds, desc="   CulturaX vi", unit="doc"):
            text = item.get("text", "").strip()
            if is_quality_paragraph(text, min_chars=200):
                paragraphs.append(text)
                collected_chars += len(text)
            if collected_chars >= target_chars * 1.2:
                break

        if paragraphs:
            print(f"   ✅ CulturaX: {len(paragraphs):,} đoạn ({collected_chars/1e6:.2f}M ký tự)")
            return paragraphs

    except Exception as e:
        print(f"   ⚠️  Lỗi CulturaX: {e}")

    # --- Fallback 2: mc4 ---
    print("\n📚 Fallback 2: Loading mc4 tiếng Việt (allenai/c4 vi)...")
    try:
        ds = load_dataset(
            "allenai/c4",
            "vi",            # subset tiếng Việt của C4 (Parquet-native)
            split="train",
            streaming=True,
        )
        paragraphs = []
        collected_chars = 0
        for item in tqdm(ds, desc="   C4 vi", unit="doc"):
            text = item.get("text", "").strip()
            if is_quality_paragraph(text, min_chars=200):
                paragraphs.append(text)
                collected_chars += len(text)
            if collected_chars >= target_chars * 1.2:
                break

        print(f"   ✅ C4-vi: {len(paragraphs):,} đoạn ({collected_chars/1e6:.2f}M ký tự)")
        return paragraphs

    except Exception as e:
        print(f"   ⚠️  Lỗi C4-vi: {e}")
        return []

# ---------------------------------------------------------------------------
# Stream builder — giống WikiText: ghép thành 1 dòng stream liên tục
# ---------------------------------------------------------------------------

def build_continuous_stream(paragraphs: list[str], target_chars: int, sep: str = "\n\n") -> str:
    """
    Ghép các đoạn văn thành một luồng văn bản liên tục.
    Cắt tại ranh giới đoạn văn gần nhất với target_chars.
    """
    stream_parts = []
    total = 0
    for para in paragraphs:
        stream_parts.append(para)
        total += len(para) + len(sep)
        if total >= target_chars:
            break

    return sep.join(stream_parts)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(output_dir: str = "./vi_bench", seed: int = 42):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🇻🇳  Vietnamese Quantization Benchmark Builder")
    print("=" * 70)
    print(f"Output dir : {out.resolve()}")
    print(f"Seed       : {seed}")
    print(f"Target     : {TOTAL_TARGET/1e6:.2f}M ký tự tổng cộng")
    print(f"Splits     : test={TARGETS['test']/1e6:.2f}M | "
          f"valid={TARGETS['valid']/1e6:.2f}M | "
          f"train={TARGETS['train']/1e6:.2f}M")
    print("=" * 70)

    # 1. Thu thập văn bản
    paragraphs = collect_from_wikipedia_vi(TOTAL_TARGET * 1.3, seed=seed)

    if len(paragraphs) < 500:
        print("\n⚠️  Wikipedia không đủ — thử CC-100...")
        paragraphs += collect_from_cc100_vi(TOTAL_TARGET * 1.3, seed=seed)

    if not paragraphs:
        raise RuntimeError("❌ Không thể tải bất kỳ nguồn dữ liệu nào!")

    # 2. Shuffle toàn bộ (seed cố định để tái hiện)
    rng = random.Random(seed)
    rng.shuffle(paragraphs)

    total_chars = sum(len(p) for p in paragraphs)
    print(f"\n📊 Tổng kho: {len(paragraphs):,} đoạn văn | {total_chars/1e6:.2f}M ký tự")

    if total_chars < TOTAL_TARGET * 0.8:
        print(f"⚠️  Cảnh báo: chỉ có {total_chars/1e6:.2f}M ký tự "
              f"(mục tiêu {TOTAL_TARGET/1e6:.2f}M). Điều chỉnh splits...")
        ratio = total_chars / TOTAL_TARGET
        for k in TARGETS:
            TARGETS[k] = int(TARGETS[k] * ratio * 0.9)

    # 3. Phân chia splits không chồng lấp
    # Tách theo số đoạn văn ước lượng
    avg_para_len = total_chars / len(paragraphs)

    n_test  = max(10, int(TARGETS["test"]  / avg_para_len))
    n_valid = max(5,  int(TARGETS["valid"] / avg_para_len))
    n_train = max(10, int(TARGETS["train"] / avg_para_len))

    # Đảm bảo không vượt quá tổng
    n_total_needed = n_test + n_valid + n_train
    if n_total_needed > len(paragraphs):
        scale = len(paragraphs) / n_total_needed
        n_test  = int(n_test  * scale)
        n_valid = int(n_valid * scale)
        n_train = len(paragraphs) - n_test - n_valid

    paras_test  = paragraphs[:n_test]
    paras_valid = paragraphs[n_test : n_test + n_valid]
    paras_train = paragraphs[n_test + n_valid : n_test + n_valid + n_train]

    splits = {
        "test" : paras_test,
        "valid": paras_valid,
        "train": paras_train,
    }

    # 4. Build streams & lưu file
    print("\n💾 Lưu splits...")
    metadata = {
        "seed": seed,
        "source": "wikipedia/20231101.vi",
        "methodology": "continuous_stream_wikitext_style",
        "splits": {},
    }

    saved_files = {}

    for split_name, paras in splits.items():
        stream = build_continuous_stream(paras, TARGETS[split_name])

        # Plain text (để đọc trực tiếp / kiểm tra)
        txt_path = out / f"vi_wiki_{split_name}.txt"
        txt_path.write_text(stream, encoding="utf-8")

        # Pickle format tương thích AWQSlidingWindowValidator
        pkl_path = out / f"vi_wiki_{split_name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump([stream], f)   # list[str] — giống format load_wikitext2_test()

        chars = len(stream)
        words = len(stream.split())

        metadata["splits"][split_name] = {
            "num_paragraphs": len(paras),
            "num_chars": chars,
            "num_words": words,
            "txt_file": txt_path.name,
            "pkl_file": pkl_path.name,
        }

        saved_files[split_name] = {"txt": txt_path, "pkl": pkl_path}

        print(f"   [{split_name:5s}] {chars/1e6:.3f}M chars | {words:,} words | "
              f"{len(paras):,} đoạn → {txt_path.name}")

    # 5. Lưu metadata
    meta_path = out / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 6. So sánh với WikiText-2 tiếng Anh
    print("\n" + "=" * 70)
    print("📊 So sánh kích thước với WikiText-2 tiếng Anh")
    print("=" * 70)
    wt2_ref = {"test": (2_088_628, 245_569), "valid": (217_646, 25_877), "train": (2_051_904, 238_854)}
    #                    chars               words
    print(f"{'Split':<8} {'VI chars':>12} {'EN chars':>12} {'VI words':>12} {'EN words':>12}")
    print("-" * 60)
    for sp in ["train", "valid", "test"]:
        vi_chars = metadata["splits"][sp]["num_chars"]
        vi_words = metadata["splits"][sp]["num_words"]
        en_chars, en_words = wt2_ref[sp]
        print(f"{sp:<8} {vi_chars:>12,} {en_chars:>12,} {vi_words:>12,} {en_words:>12,}")

    print("\n✅ Dataset hoàn chỉnh!")
    print(f"   📂 Output: {out.resolve()}")
    print(f"   📄 Metadata: {meta_path}")

    return saved_files, metadata


# ---------------------------------------------------------------------------
# Integration loader — dùng trực tiếp trong AWQSlidingWindowValidator
# ---------------------------------------------------------------------------

def load_vi_wiki_split(split: str = "test", data_dir: str = "./vi_bench") -> list[str]:
    """
    Drop-in replacement cho load_wikitext2_test() trong AWQSlidingWindowValidator.

    Ví dụ:
        texts = load_vi_wiki_split("test", "./vi_bench")
        results = validator.evaluate_sliding_window(model, tokenizer, texts)
    """
    pkl_path = Path(data_dir) / f"vi_wiki_{split}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {pkl_path}. Chạy build_dataset() trước."
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Vietnamese quantization evaluation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="./vi_bench",
                        help="Thư mục lưu dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed để tái hiện")
    args = parser.parse_args()

    build_dataset(output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()