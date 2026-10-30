"""Clean music metadata and create reproducible train/test lists.

The input CSV has three columns: file_name, file_path, genre.
For a full audio check, put tracks in data/genres_original/<genre>/<file>.
Use --metadata-only if the audio is not available yet (a preliminary check).
"""

import argparse
import csv
import hashlib
import json
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path

SUPPORTED = {".wav", ".mp3", ".flac", ".ogg"}
COLUMNS = ["file_name", "file_path", "genre"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_metadata(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError("CSV must contain: file_name, file_path, genre")
        return list(reader)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_wav(path):
    # Standard PCM WAV validation; other formats only get a nonempty-file check.
    if path.suffix.lower() != ".wav":
        return True
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() > 0 and audio.getframerate() > 0
    except (wave.Error, EOFError, OSError):
        return False


def clean_rows(rows, audio_root, metadata_only=False):
    valid = []
    discarded = []
    seen_names = set()
    root = Path(audio_root)
    for index, source in enumerate(rows, start=2):
        name = (source.get("file_name") or "").strip()
        genre = (source.get("genre") or "").strip().lower()
        if not name or not genre:
            discarded.append({"csv_row": index, "reason": "missing_filename_or_genre"})
            continue
        if Path(name).name != name or Path(name).suffix.lower() not in SUPPORTED:
            discarded.append({"csv_row": index, "reason": "invalid_name_or_extension"})
            continue
        # Build paths relative to the current dataset directory.
        path = root / genre / name
        identity = (genre, name.lower())
        if identity in seen_names:
            discarded.append({"csv_row": index, "reason": "duplicate_metadata_row"})
            continue
        seen_names.add(identity)
        if not metadata_only:
            if not path.is_file():
                discarded.append({"csv_row": index, "reason": "missing_audio_file"})
                continue
            if path.stat().st_size == 0 or not validate_wav(path):
                discarded.append({"csv_row": index, "reason": "empty_or_invalid_audio"})
                continue
        # Relative POSIX paths are portable inside the repository.
        try:
            portable_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            portable_path = path.as_posix()
        record = {"file_name": name, "file_path": portable_path, "genre": genre}
        if not metadata_only:
            record["_digest"] = file_digest(path)
        valid.append(record)

    # With real files, prevent identical audio from appearing on either side of split.
    if not metadata_only:
        by_hash = defaultdict(list)
        for record in valid:
            by_hash[record["_digest"]].append(record)
        kept = []
        for same_audio in by_hash.values():
            if len({r["genre"] for r in same_audio}) > 1:
                for record in same_audio:
                    discarded.append({"file_name": record["file_name"], "reason": "conflicting_genre_labels_for_same_audio"})
            else:
                kept.append(same_audio[0])
                for record in same_audio[1:]:
                    discarded.append({"file_name": record["file_name"], "reason": "duplicate_audio_content"})
        valid = kept
    for record in valid:
        record.pop("_digest", None)
    return valid, discarded


def split_data(records, ratio=0.2, seed=42):
    if not 0 < ratio < 1:
        raise ValueError("test ratio must be between 0 and 1")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[record["genre"]].append(record)
    train, test, small_classes = [], [], []
    for genre in sorted(groups):
        items = sorted(groups[genre], key=lambda item: item["file_path"])
        rng.shuffle(items)
        if len(items) < 2:
            train.extend(items)
            small_classes.append(genre)
            continue
        n_test = min(len(items) - 1, max(1, round(len(items) * ratio)))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    train.sort(key=lambda row: (row["genre"], row["file_name"]))
    test.sort(key=lambda row: (row["genre"], row["file_name"]))
    assert {r["file_path"] for r in train}.isdisjoint({r["file_path"] for r in test})
    return train, test, small_classes


def save_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def evaluate(metadata, audio_root, output, ratio, seed, metadata_only):
    rows = read_metadata(metadata)
    clean, discarded = clean_rows(rows, audio_root, metadata_only)
    train, test, small_classes = split_data(clean, ratio, seed)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "clean_metadata.csv", clean)
    save_csv(output / "train.csv", train)
    save_csv(output / "test.csv", test)
    warnings = []
    if metadata_only:
        warnings.append("Audio files were not checked: metadata-only mode.")
    if small_classes:
        warnings.append("Genres absent from test (only one sample): " + ", ".join(small_classes))
    if len(test) == 0 or len({r["genre"] for r in test}) < 2:
        warnings.append("Too little test data for meaningful classification metrics.")
    if not metadata_only:
        warnings.append("Non-WAV file integrity was checked by file size, not full decoding.")
    warnings.append("Artist/album leakage could not be checked without artist/album metadata.")
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(clean),
        "discarded_rows": len(discarded),
        "train_rows": len(train),
        "test_rows": len(test),
        "genre_counts": dict(sorted(Counter(x["genre"] for x in clean).items())),
        "train_genre_counts": dict(sorted(Counter(x["genre"] for x in train).items())),
        "test_genre_counts": dict(sorted(Counter(x["genre"] for x in test).items())),
        "test_coverage_complete": len({x["genre"] for x in clean}) == len({x["genre"] for x in test}),
        "discarded": discarded,
        "mode": "metadata_only" if metadata_only else "audio_checked",
        "seed": seed,
        "requested_test_ratio": ratio,
        "warnings": warnings,
    }
    (output / "quality_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Input:", len(rows), "| Clean:", len(clean), "| Train:", len(train), "| Test:", len(test))
    print("Genres:", summary["genre_counts"])
    print("Test genres:", summary["test_genre_counts"])
    for message in warnings:
        print("WARNING:", message)
    print("Saved CSV files and quality_report.json in", output)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare and evaluate music genre metadata")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--audio-root", type=Path, default=PROJECT_ROOT / "data/genres_original")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if not args.metadata.is_file():
        parser.error(f"Metadata CSV not found: {args.metadata}. Generate metadata.csv with load_data.py before running this script")
    evaluate(args.metadata, args.audio_root, args.output, args.test_ratio, args.seed, args.metadata_only)


if __name__ == "__main__":
    main()
