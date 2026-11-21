import argparse

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=".")
    parser.add_argument('--revision',default='main')
    args=parser.parse_args(argv)
    from pathlib import Path
    import hashlib
    import json

    from datasets import load_dataset
    from huggingface_hub import HfApi

    # ============================================================
    # Settings
    # ============================================================

    ROOT = Path(args.root).resolve()
    DEST = ROOT / "data" / "race"
    HF_CACHE = ROOT / "data" / "hf_cache"

    DATASET_ID = "ehovy/race"
    CONFIG = "high"

    DEST.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    # Record the exact dataset revision for reproducibility.
    revision = HfApi().dataset_info(DATASET_ID,revision=args.revision).sha
    print("Dataset:", DATASET_ID, "| subset:", CONFIG)
    print("Revision:", revision)

    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        revision=revision,
        cache_dir=str(HF_CACHE),
    )

    # ============================================================
    # Convert to comparison-friendly JSON records
    # ============================================================

    def text_hash(text):
        normalized = " ".join(text.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


    def make_record(row, split, index):
        article = row["article"]
        question_text = row["question"]
        options = list(row["options"])
        gold = row["answer"].strip().upper()

        if len(options) != 4 or gold not in "ABCD" or len(gold) != 1:
            raise ValueError(f"Invalid answer/options: {split}, row {index}")

        # This is the full question supplied to the model.
        # The correct answer is NOT included.
        formatted_question = (
            f"Passage:\n{article}\n\n"
            f"Question:\n{question_text}\n\n"
            "Options:\n"
            + "\n".join(
                f"{letter}. {option}"
                for letter, option in zip("ABCD", options)
            )
        )

        return {
            "question_id": text_hash(formatted_question),
            "dataset_idx": index,
            "dataset": "RACE-H",
            "split": split,
            "source_example_id": row["example_id"],
            # Questions sharing a passage should share an OOF fold.
            "group_id": text_hash(article),
            "question": formatted_question,
            "article": article,
            "question_text": question_text,
            "options": options,
            "gold": gold,
        }


    expected_counts = {
        "train": 62445,
        "validation": 3451,
        "test": 3498,
    }

    manifest = {
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "revision": revision,
        "answer_type": "multiple_choice_letter",
        "splits": {},
    }

    for split, expected in expected_counts.items():
        rows = [
            make_record(row, split, index)
            for index, row in enumerate(dataset[split])
        ]

        if len(rows) != expected:
            raise RuntimeError(
                f"Unexpected {split} size: {len(rows)}; expected {expected}."
            )

        path = DEST / f"race_high_{split}.json"
        if path.exists() and json.loads(path.read_text())!=rows:raise FileExistsError(f'Existing data differ: {path}; choose another root')
        temporary = path.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

        manifest["splits"][split] = {
            "questions": len(rows),
            "passage_groups": len({row["group_id"] for row in rows}),
            "file": path.name,
        }

        print(
            f"{split:10s}: {len(rows):,} questions | "
            f"{manifest['splits'][split]['passage_groups']:,} passages "
            f"| saved: {path}"
        )

    (DEST / "race_high_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nExample model input:\n")
    print(make_record(dataset["test"][0], "test", 0)["question"])
    print("\nGold:", dataset["test"][0]["answer"])

if __name__=="__main__":main()
