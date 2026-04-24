"""
Tích hợp Vietnamese benchmark vào AWQSlidingWindowValidator
============================================================
Thêm phương thức load_vi_wiki() và đăng ký dataset "ViWiki" vào run_validation().

Dán đoạn này vào cuối class AWQSlidingWindowValidator (hoặc import như mixin).
"""

import pickle
from pathlib import Path


# ---------------------------------------------------------------------------
# Mixin — thêm vào AWQSlidingWindowValidator
# ---------------------------------------------------------------------------

class VietnameseBenchmarkMixin:
    """
    Mixin thêm khả năng đánh giá dataset tiếng Việt.
    Dùng:
        class AWQSlidingWindowValidator(VietnameseBenchmarkMixin, ...):
            ...
    """

    def load_vi_wiki(self, split: str = "test", vi_bench_dir: str = "./vi_bench") -> list[str]:
        """
        Load Vietnamese Wikipedia benchmark (split = test | valid | train).

        Trả về list[str] — format giống load_wikitext2_test().

        Tham số:
            split         : "test", "valid", hoặc "train"
            vi_bench_dir  : thư mục chứa output của build_vi_benchmark.py
        """
        print(f"\n[VI] Loading Vietnamese Wikipedia benchmark ({split})...")

        # Thử load từ cache pkl trước
        pkl_path = Path(vi_bench_dir) / f"vi_wiki_{split}.pkl"
        if pkl_path.exists():
            print(f"  📦 Loading from cache: {pkl_path}")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            chars = sum(len(t) for t in data)
            print(f"  ✅ Loaded {chars/1e6:.2f}M chars")
            return data

        # Nếu chưa có pkl, thử txt
        txt_path = Path(vi_bench_dir) / f"vi_wiki_{split}.txt"
        if txt_path.exists():
            print(f"  📄 Loading from txt: {txt_path}")
            text = txt_path.read_text(encoding="utf-8")
            data = [text]
            # Cache lại
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
            print(f"  ✅ Loaded {len(text)/1e6:.2f}M chars (cached to pkl)")
            return data

        raise FileNotFoundError(
            f"Không tìm thấy {pkl_path} hay {txt_path}.\n"
            f"Chạy build_vi_benchmark.py --output-dir {vi_bench_dir} trước."
        )


# ---------------------------------------------------------------------------
# Patch hàm run_validation để thêm ViWiki vào danh sách datasets
# ---------------------------------------------------------------------------

def run_validation_with_vi(
    validator,
    heuristic_path: str,
    standard_path: str = None,
    n_samples: int = 2000,
    vi_bench_dir: str = "./vi_bench",
    include_vi: bool = True,
):
    """
    Drop-in replacement cho validator.run_validation() có thêm ViWiki.

    Ví dụ:
        validator = AWQSlidingWindowValidator(cache_dir="./dataset_cache")
        run_validation_with_vi(
            validator,
            heuristic_path="/path/to/heuristic_awq",
            standard_path="/path/to/standard_awq",
            vi_bench_dir="./vi_bench",
        )
    """
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    datasets = {
        "WikiText-2": validator.load_wikitext2_test(n_samples),
        "C4":         validator.load_c4_validation(n_samples),
    }

    if include_vi:
        try:
            # Thêm ViWiki mixin nếu chưa có
            if not hasattr(validator, "load_vi_wiki"):
                # Monkey-patch tại runtime
                import types
                validator.load_vi_wiki = types.MethodType(
                    VietnameseBenchmarkMixin.load_vi_wiki, validator
                )

            datasets["ViWiki"] = validator.load_vi_wiki("test", vi_bench_dir)
            print("✅ ViWiki test split loaded")
        except FileNotFoundError as e:
            print(f"⚠️  Bỏ qua ViWiki: {e}")

    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)

    models = {"Heuristic AWQ": heuristic_path}
    if standard_path:
        models["Standard AWQ"] = standard_path

    for dataset_name, texts in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        for model_name, model_path in models.items():
            result = validator.evaluate_model_on_dataset(
                model_path, model_name, texts, dataset_name
            )
            if result:
                if dataset_name not in validator.results:
                    validator.results[dataset_name] = {}
                validator.results[dataset_name][model_name] = result

    return validator.results


# ---------------------------------------------------------------------------
# Standalone test — kiểm tra dataset sau khi build
# ---------------------------------------------------------------------------

def inspect_vi_bench(vi_bench_dir: str = "./vi_bench"):
    """In thống kê nhanh về bộ dữ liệu đã build."""
    import json

    vi_bench = Path(vi_bench_dir)
    meta_path = vi_bench / "metadata.json"

    if not meta_path.exists():
        print(f"❌ Không tìm thấy {meta_path}")
        return

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    print("=" * 60)
    print("📊 Vietnamese Benchmark — Thống kê")
    print("=" * 60)
    print(f"Source : {meta['source']}")
    print(f"Seed   : {meta['seed']}")
    print(f"Method : {meta['methodology']}")
    print()

    # WikiText-2 English reference
    wt2_ref = {
        "test":  {"num_chars": 2_088_628, "num_words": 245_569},
        "valid": {"num_chars":   217_646, "num_words":  25_877},
        "train": {"num_chars": 2_051_904, "num_words": 238_854},
    }

    print(f"{'Split':<8} {'VI Chars':>12} {'EN Chars':>12}  {'VI Words':>10} {'EN Words':>10}  {'Ratio':>7}")
    print("-" * 65)
    for sp in ["train", "valid", "test"]:
        s = meta["splits"][sp]
        ref = wt2_ref[sp]
        ratio = s["num_chars"] / ref["num_chars"]
        print(f"{sp:<8} {s['num_chars']:>12,} {ref['num_chars']:>12,}  "
              f"{s['num_words']:>10,} {ref['num_words']:>10,}  {ratio:>7.2f}x")

    print()
    print("Files:")
    for sp, s in meta["splits"].items():
        print(f"  [{sp}] {s['txt_file']}  ({s['num_chars']/1e6:.2f}M chars)")
        print(f"        {s['pkl_file']}")

    # Preview 3 đoạn đầu của test split
    test_pkl = vi_bench / "vi_wiki_test.pkl"
    if test_pkl.exists():
        with open(test_pkl, "rb") as f:
            stream = pickle.load(f)[0]
        preview_paras = stream.split("\n\n")[:3]
        print("\n📝 Preview (3 đoạn đầu test split):")
        print("-" * 60)
        for i, para in enumerate(preview_paras, 1):
            print(f"[{i}] {para[:200]}{'...' if len(para)>200 else ''}")
            print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        inspect_vi_bench(sys.argv[1])
    else:
        inspect_vi_bench()