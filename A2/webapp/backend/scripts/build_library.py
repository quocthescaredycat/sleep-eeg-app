"""Build the similarity-search reference library from Sleep-EDF (Phase 1).

    python scripts/build_library.py extract --edf-dir DIR --labels sleep_waveform.csv
    python scripts/build_library.py index [--recreate]   # Qdrant collection from the files
    python scripts/build_library.py load-db              # library_epochs table (PostgreSQL)
    python scripts/build_library.py verify               # files, Qdrant and PostgreSQL agree

``extract`` runs the serving code path on every night: ``preprocess_recording`` for the
signal and ``parse_hypnogram`` for the expert stages. It stops unless the benchmark epochs
it finds are exactly the rows of the A2 label file, with the same stages, so the library
and the A2 results describe the same 195,469 epochs. Each night lands in ``parts/`` as soon
as it is done; an interrupted run resumes where it stopped.

Inference runs on CPU with one thread per process: these networks are too small for
intra-op threading to pay off, and CPU is what the worker uses, so library vectors and
query vectors come from the same kernels.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.library import store  # noqa: E402
from app.ml.preprocessing import (  # noqa: E402
    EPOCH_SECONDS,
    PREPROCESSING_VERSION,
    preprocess_recording,
)
from app.ml.registry import CLASS_NAMES, load_releases, sha256  # noqa: E402
from app.ml.signal_features import (  # noqa: E402
    FEATURE_NAMES,
    SIGNAL_FEATURES_VERSION,
    signal_features,
)
from app.services.edf import parse_hypnogram  # noqa: E402

DISPLAY_SAMPLES = EPOCH_SECONDS * 100  # the 100 Hz display waveform of one epoch
LABEL_COLUMNS = {"participant", "recording", "onset", "label", "split", "validation_fold"}


class ParityError(RuntimeError):
    """The serving pipeline does not reproduce the A2 label file for a night."""


def read_labels(path: Path) -> dict[str, list[dict]]:
    """Rows of the A2 label file grouped by night, in file order; ``id`` is the row number."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = LABEL_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"{path.name} lacks columns {sorted(missing)}.")
        nights: dict[str, list[dict]] = {}
        for number, row in enumerate(reader):
            row["id"] = number
            nights.setdefault(row["recording"], []).append(row)
    ids = [row["id"] for rows in nights.values() for row in rows]
    if ids != list(range(len(ids))):
        raise SystemExit(f"{path.name}: each night's rows must be contiguous.")
    return nights


def only(paths: list[Path], what: str) -> Path:
    if len(paths) != 1:
        raise SystemExit(f"Expected exactly one {what}, found {len(paths)}.")
    return paths[0]


# --- extract: one worker process per night ---

_MODELS: dict[str, list] = {}


def _init_worker(checkpoints_dir: str) -> None:
    import torch

    from app.ml.inference import load_ensemble

    torch.set_num_threads(1)
    releases = load_releases(checkpoints_dir)
    for space in store.MODEL_SPACES:
        _MODELS[space] = load_ensemble(releases[store.SPACES[space].release])


def extract_night(recording: str, psg: str, hypnogram: str, rows: list[dict], fold: int,
                  parts: str) -> dict:
    from app.ml.embeddings import embed_and_classify

    started = time.perf_counter()
    prepared = preprocess_recording(psg)
    stages = parse_hypnogram(Path(hypnogram), prepared.n_epochs)
    benchmark = sorted(index for index, (_, inside) in stages.items() if inside)
    expected = [int(float(row["onset"])) // EPOCH_SECONDS for row in rows]
    if benchmark != expected:
        raise ParityError(f"{recording}: the app finds {len(benchmark)} benchmark epochs, "
                          f"the label file lists {len(expected)}.")
    wrong = [index for index, row in zip(expected, rows, strict=True)
             if stages[index][0] != int(row["label"])]
    if wrong:
        raise ParityError(f"{recording}: {len(wrong)} epochs disagree with the label file, "
                          f"first at epoch {wrong[0]}.")

    index = np.asarray(expected)
    epochs = prepared.epochs[index]
    display = prepared.display_waveform()[:, :prepared.n_epochs * DISPLAY_SAMPLES]
    waveforms = display.reshape(2, prepared.n_epochs, DISPLAY_SAMPLES).transpose(1, 0, 2)
    waveforms = waveforms[index].astype(np.float16)
    if not np.isfinite(waveforms).all():
        raise ParityError(f"{recording}: display waveform overflows float16.")
    arrays = {"ids": np.array([row["id"] for row in rows], dtype=np.int64),
              "waveforms": waveforms, "signal": signal_features(epochs)}
    for space, models in _MODELS.items():
        probabilities = []
        for number, model in enumerate(models, start=1):
            features, probs = embed_and_classify(model, epochs)
            probabilities.append(probs)
            if number == fold:
                arrays[f"vectors_{space}"] = features
        arrays[f"probabilities_{space}"] = np.stack(probabilities)

    partial = Path(parts) / f"{recording}.partial.npz"
    np.savez(partial, **arrays)
    partial.replace(Path(parts) / f"{recording}.npz")
    return {"recording": recording, "epochs": len(index),
            "seconds": time.perf_counter() - started}


def check_releases(checkpoints: Path, fold: int) -> dict:
    releases = load_releases(checkpoints)
    summary = {}
    for space in store.MODEL_SPACES:
        release = releases.get(store.SPACES[space].release)
        if release is None or not release.available:
            raise SystemExit(f"{store.SPACES[space].release} is not installed in {checkpoints} "
                             "(scripts/checkpoints.py package).")
        if release.synthetic:
            raise SystemExit(f"{release.id} has synthetic weights: a library needs trained ones.")
        if release.preprocessing.get("version") != PREPROCESSING_VERSION:
            raise SystemExit(f"{release.id} was trained with another preprocessing version.")
        release.verify()
        if release.folds[fold - 1]["file"] != f"fold_{fold}.pt":
            raise SystemExit(f"{release.id}: fold order differs from fold_1..fold_5.")
        summary[release.id] = {"embedding_fold_file": f"fold_{fold}.pt",
                               "folds_sha256": [entry["sha256"] for entry in release.folds]}
    return summary


def extract(args: argparse.Namespace) -> int:
    nights = read_labels(args.labels)
    releases = check_releases(args.checkpoints, args.fold)
    parts = args.dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    tasks = []
    for recording, rows in nights.items():
        if (parts / f"{recording}.npz").is_file():
            continue
        psg = only(sorted(args.edf_dir.glob(f"{recording}*-PSG.edf")), f"{recording} PSG")
        hypnogram = only(sorted(args.edf_dir.glob(f"{recording}*-Hypnogram.edf")),
                         f"{recording} hypnogram")
        tasks.append((recording, str(psg), str(hypnogram), rows))
    tasks.sort(key=lambda task: -len(task[3]))  # longest nights first balances the pool
    total_epochs = sum(len(task[3]) for task in tasks)
    print(f"{len(nights)} nights, {len(nights) - len(tasks)} already extracted; "
          f"{len(tasks)} to go ({total_epochs} epochs) on {args.workers} processes", flush=True)

    started, done_epochs = time.perf_counter(), 0
    if tasks:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                 initargs=(str(args.checkpoints),)) as pool:
            futures = [pool.submit(extract_night, recording, psg, hypnogram, rows, args.fold,
                                   str(parts))
                       for recording, psg, hypnogram, rows in tasks]
            try:
                for done, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    done_epochs += result["epochs"]
                    elapsed = time.perf_counter() - started
                    remaining = elapsed / done_epochs * (total_epochs - done_epochs)
                    print(f"[{done}/{len(tasks)}] {result['recording']}: {result['epochs']} "
                          f"epochs in {result['seconds']:.0f} s | elapsed {elapsed / 60:.1f} min,"
                          f" ~{remaining / 60:.1f} min left", flush=True)
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise
    assemble(args, nights, releases, time.perf_counter() - started)
    if not args.keep_parts:
        shutil.rmtree(parts)
    return 0


def assemble(args: argparse.Namespace, nights: dict[str, list[dict]], releases: dict,
             seconds: float) -> None:
    """Concatenate the per-night parts into the library files, in label-file order."""
    rows = [row for night in nights.values() for row in night]
    count = len(rows)
    shapes = {"waveforms": ((count, 2, DISPLAY_SAMPLES), np.float16, "waveforms_100hz.npy"),
              "signal": ((count, len(FEATURE_NAMES)), np.float32, "features_signal.npy")}
    for space in store.MODEL_SPACES:
        shapes[f"vectors_{space}"] = ((count, store.SPACES[space].dim), np.float32,
                                      store.SPACES[space].file)
    outputs = {key: np.lib.format.open_memmap(args.dir / f"{name}.partial", mode="w+",
                                              dtype=dtype, shape=shape)
               for key, (shape, dtype, name) in shapes.items()}
    probabilities = {space: np.zeros((5, count, 5), dtype=np.float32)
                     for space in store.MODEL_SPACES}
    for recording, night in nights.items():
        with np.load(args.dir / "parts" / f"{recording}.npz") as part:
            ids = part["ids"]
            if not np.array_equal(ids, [row["id"] for row in night]):
                raise SystemExit(f"parts/{recording}.npz does not match the label file.")
            for key, output in outputs.items():
                output[ids] = part[key]
            for space in store.MODEL_SPACES:
                probabilities[space][:, ids] = part[f"probabilities_{space}"]
    signal = np.asarray(outputs["signal"], dtype=np.float64)
    for output in outputs.values():
        output.flush()
    del outputs
    for _, _, name in shapes.values():
        (args.dir / f"{name}.partial").replace(args.dir / name)
    np.savez(args.dir / "probabilities.npz", **probabilities)

    with (args.dir / "epochs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(store.EPOCH_COLUMNS)
        for row in rows:
            writer.writerow([row["id"], row["participant"], row["recording"],
                             int(float(row["onset"])) // EPOCH_SECONDS, row["label"],
                             row["split"], row["validation_fold"]])

    import mne
    import torch

    stages = Counter(int(row["label"]) for row in rows)
    unseen = sum(row["split"] == "test" or int(row["validation_fold"]) == args.fold
                 for row in rows)
    files = ["epochs.csv", "probabilities.npz", *(name for _, _, name in shapes.values())]
    manifest = {
        "library_id": store.LIBRARY_ID,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fold": args.fold,
        "n_epochs": count,
        "n_participants": len({row["participant"] for row in rows}),
        "n_recordings": len(nights),
        "class_counts": {CLASS_NAMES[stage]: stages[stage] for stage in range(len(CLASS_NAMES))},
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "unseen_by_embedding_fold": unseen,
        "preprocessing": PREPROCESSING_VERSION,
        "signal_features": {"version": SIGNAL_FEATURES_VERSION, "names": list(FEATURE_NAMES),
                            "mean": signal.mean(axis=0).round(6).tolist(),
                            "std": signal.std(axis=0).round(6).tolist()},
        "spaces": {name: {"dim": space.dim, "distance": space.distance.value,
                          "release": space.release, "file": space.file}
                   for name, space in store.SPACES.items()},
        "releases": releases,
        "labels": {"file": args.labels.name, "sha256": sha256(args.labels)},
        "files": {name: {"bytes": (args.dir / name).stat().st_size,
                         "sha256": sha256(args.dir / name)} for name in files},
        "build": {"processes": args.workers, "extract_seconds": round(seconds, 1),
                  "torch": torch.__version__, "mne": mne.__version__,
                  "numpy": np.__version__},
    }
    (args.dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Library written to {args.dir}: {count} epochs, {unseen} unseen by fold {args.fold}.")


# --- index, load-db, verify ---

def connect():
    settings = get_settings()
    return store.connect(settings.QDRANT_URL, settings.QDRANT_GRPC_PORT)


def wait_until_green(client, name: str, what: str, timeout: float = 3600) -> None:
    from qdrant_client import models

    deadline = time.monotonic() + timeout
    time.sleep(2)  # an optimisation just requested may not have started yet
    while True:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN:
            return
        if time.monotonic() > deadline:
            raise SystemExit(f"{name}: still {info.status} after {timeout:.0f} s ({what}).")
        print(f"  {what}: status {info.status.value}, "
              f"indexed vectors {info.indexed_vectors_count}", flush=True)
        time.sleep(10)


def index(args: argparse.Namespace) -> int:
    manifest = store.load_manifest(args.dir)
    epochs = store.load_epochs(args.dir)
    name = store.collection_name(manifest)
    client = connect()
    if client.collection_exists(name):
        if not args.recreate:
            raise SystemExit(f"Collection {name} exists; pass --recreate to rebuild it.")
        client.delete_collection(name)
    store.create_collection(client, name)

    vectors = {space: np.load(args.dir / store.SPACES[space].file, mmap_mode="r")
               for space in store.SPACES}
    count = manifest["n_epochs"]
    started = time.perf_counter()
    from qdrant_client import models

    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        batch = {space: np.asarray(array[start:stop], dtype=np.float32)
                 for space, array in vectors.items()}
        batch["signal"] = store.standardize_signal(batch["signal"], manifest)
        client.upsert(name, wait=False, points=models.Batch(
            ids=list(range(start, stop)),
            vectors={space: array.tolist() for space, array in batch.items()},
            payloads=[store.payload(epochs, row) for row in range(start, stop)]))
        if (stop // args.batch_size) % 40 == 0 or stop == count:
            print(f"  uploaded {stop}/{count} points "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
    wait_until_green(client, name, "upload")
    store.build_hnsw(client, name)
    wait_until_green(client, name, "HNSW build")
    info = client.get_collection(name)
    print(f"{name}: {info.points_count} points, {info.indexed_vectors_count} indexed vectors, "
          f"{time.perf_counter() - started:.0f} s")
    return 0


def library_rows(directory: Path) -> list[dict]:
    manifest = store.load_manifest(directory)
    epochs = store.load_epochs(directory)
    with np.load(directory / "probabilities.npz") as stored:
        predictions = {space: store.out_of_fold(stored[space], epochs["split"],
                                                epochs["validation_fold"])
                       for space in store.MODEL_SPACES}
    seen = store.embedding_training_mask(epochs, manifest["fold"])
    rows = []
    for row in range(len(epochs["id"])):
        fold = int(epochs["validation_fold"][row])
        entry = {"id": row, "participant": str(epochs["participant"][row]),
                 "recording": str(epochs["recording"][row]),
                 "epoch_index": int(epochs["epoch_index"][row]),
                 "stage": int(epochs["stage"][row]), "split": str(epochs["split"][row]),
                 "validation_fold": None if fold == store.TEST_FOLD else fold,
                 "in_embedding_training": bool(seen[row])}
        for space, probs in predictions.items():
            entry[f"{space}_stage"] = int(probs[row].argmax())
            entry[f"{space}_confidence"] = round(float(probs[row].max()), 5)
        rows.append(entry)
    return rows


def load_db(args: argparse.Namespace) -> int:
    from sqlalchemy import insert, inspect, text

    from app.core.db import get_engine
    from app.models import LibraryEpoch

    engine = get_engine()
    if not inspect(engine).has_table(LibraryEpoch.__tablename__):
        raise SystemExit("Table library_epochs is missing: run `alembic upgrade head` first.")
    rows = library_rows(args.dir)
    with engine.begin() as connection:  # one transaction: readers never see half a library
        connection.execute(text("TRUNCATE library_epochs"))
        for start in range(0, len(rows), 5000):
            connection.execute(insert(LibraryEpoch), rows[start:start + 5000])
    print(f"library_epochs: {len(rows)} rows")
    return 0


def verify(args: argparse.Namespace) -> int:
    from sqlalchemy import func, select

    from app.core.db import get_sessionmaker
    from app.models import LibraryEpoch

    manifest = store.load_manifest(args.dir)
    epochs = store.load_epochs(args.dir)
    count = manifest["n_epochs"]
    problems = []
    for name, entry in manifest["files"].items():
        path = args.dir / name
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            problems.append(f"{name}: missing or wrong size")
        elif args.hash and sha256(path) != entry["sha256"]:
            problems.append(f"{name}: SHA-256 mismatch")
    vectors = {space: np.load(args.dir / store.SPACES[space].file, mmap_mode="r")
               for space in store.SPACES}
    for space, array in vectors.items():
        if array.shape != (count, store.SPACES[space].dim):
            problems.append(f"{store.SPACES[space].file}: shape {array.shape}")

    client = connect()
    name = store.collection_name(manifest)
    info = client.get_collection(name)
    if info.points_count != count:
        problems.append(f"Qdrant {name}: {info.points_count} points, expected {count}")
    sample = np.random.default_rng(0).choice(count, size=min(200, count), replace=False)
    points = client.retrieve(name, ids=[int(i) for i in sample], with_vectors=True,
                             with_payload=True)
    for point in points:
        row = int(point.id)
        if point.payload != store.payload(epochs, row):
            problems.append(f"Qdrant point {row}: payload differs from epochs.csv")
        for space in store.MODEL_SPACES:
            stored, source = np.asarray(point.vector[space]), np.asarray(vectors[space][row])
            cosine = stored @ source / np.linalg.norm(stored) / np.linalg.norm(source)
            if cosine < 0.9999:
                problems.append(f"Qdrant point {row}: {space} vector differs (cos {cosine:.5f})")
        expected = store.standardize_signal(np.asarray(vectors["signal"][row]), manifest)
        if not np.allclose(point.vector["signal"], expected, atol=1e-4):
            problems.append(f"Qdrant point {row}: signal vector differs")
    if len(points) != len(sample):
        problems.append(f"Qdrant returned {len(points)} of {len(sample)} sampled points")

    with get_sessionmaker()() as session:
        rows = session.scalar(select(func.count()).select_from(LibraryEpoch))
        if rows != count:
            problems.append(f"library_epochs: {rows} rows, expected {count}")
        for row in sample[:50]:
            entry = session.get(LibraryEpoch, int(row))
            if entry is None or entry.stage != int(epochs["stage"][row]):
                problems.append(f"library_epochs id {row}: missing or wrong stage")

    for problem in problems:
        print("FAIL", problem)
    print("OK: files, Qdrant and PostgreSQL agree" if not problems
          else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "index", "load-db", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--dir", type=Path, default=settings.LIBRARY_DIR)
    extract_command = commands.choices["extract"]
    extract_command.add_argument("--edf-dir", type=Path, required=True,
                                 help="Sleep-EDF sleep-cassette directory")
    extract_command.add_argument("--labels", type=Path, required=True,
                                 help="A2 label file, e.g. dataset/sleep_waveform.csv")
    extract_command.add_argument("--checkpoints", type=Path, default=settings.CHECKPOINTS_DIR)
    extract_command.add_argument("--fold", type=int, choices=range(1, 6), default=5,
                                 help="fold whose features span the model spaces")
    extract_command.add_argument("--workers", type=int, default=10)
    extract_command.add_argument("--keep-parts", action="store_true")
    commands.choices["index"].add_argument("--recreate", action="store_true")
    commands.choices["index"].add_argument("--batch-size", type=int, default=256)
    commands.choices["verify"].add_argument("--hash", action="store_true",
                                            help="also re-hash every file (slow)")
    args = parser.parse_args()
    handler = {"extract": extract, "index": index, "load-db": load_db, "verify": verify}
    return handler[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
