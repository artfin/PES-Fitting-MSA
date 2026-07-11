#!/usr/bin/env python3
"""Migrate a flat model folder to per-run directories (Phase 3 of run-tracking).

Folds the prefix-sibling files of each run into ``<folder>/runs/<stem>/``:

    models/h2o-h2o/water-extended-72-sharded.yaml   ->  runs/water-extended-72-sharded/water-extended-72-sharded.yaml
    models/h2o-h2o/water-extended-72-sharded.log    ->  runs/.../water-extended-72-sharded.log
    models/h2o-h2o/water-extended-72-sharded-test.* ->  runs/.../test/...

Filenames are kept stem-based (not renamed to config.yaml/train.log), so
``train_model.py`` / ``eval_model.py`` keep working when pointed at the run dir
with ``--model_name=<stem>`` -- no training-code changes needed.

Runs are identified by their ``<stem>.yaml`` (excluding ``*-test.yaml``). Files
are matched to the *longest* owning stem, so ``water-extended-7-grad.log`` goes
to that run and not to ``water-extended``. Files owned by no run (README, golden,
report.html, orphan .png/.pt without a config) are left in place.

A ``.migration_manifest.json`` records every move for a clean ``--reverse``.

Usage:
    python src/run_tracking/migrate.py --folder models/h2o-h2o --dry-run
    python src/run_tracking/migrate.py --folder models/h2o-h2o
    python src/run_tracking/migrate.py --folder models/h2o-h2o --reverse
"""

import argparse
import json
import os
import shutil

MANIFEST = ".migration_manifest.json"

# Files at the folder root that must never be treated as run artifacts.
_KEEP = {"README", "golden", "report.html", MANIFEST}


def run_stems(folder):
    """Return the set of run identifiers (``<stem>.yaml`` minus ``*-test.yaml``)."""
    stems = set()
    for name in os.listdir(folder):
        if name.endswith(".yaml"):
            stem = name[:-len(".yaml")]
            if not stem.endswith("-test"):
                stems.add(stem)
    return stems


def _owner(name, stems_by_len):
    """Return (stem, is_test) for a filename, or (None, False) if unowned."""
    for stem in stems_by_len:
        if name.startswith(stem):
            rest = name[len(stem):]
            if rest and rest[0] in ".-":
                # Eval sidecars are '<stem>...-test.*' -- match '-test' anywhere
                # in the suffix, so '<stem>-grad-test.log' also lands in test/.
                return stem, "-test" in rest
    return None, False


def plan_moves(folder):
    """Return a list of (src_abs, dst_abs, src_rel, dst_rel) moves."""
    stems = run_stems(folder)
    stems_by_len = sorted(stems, key=len, reverse=True)
    runs_root = os.path.join(folder, "runs")

    moves = []
    for name in sorted(os.listdir(folder)):
        src = os.path.join(folder, name)
        if not os.path.isfile(src):
            continue  # skip subdirs (incl. an existing runs/)
        if name in _KEEP:
            continue

        stem, is_test = _owner(name, stems_by_len)
        if stem is None:
            continue  # orphan / folder-level file -> leave in place

        dst_dir = os.path.join(runs_root, stem, "test") if is_test \
            else os.path.join(runs_root, stem)
        dst = os.path.join(dst_dir, name)
        moves.append((
            src, dst,
            os.path.relpath(src, folder),
            os.path.relpath(dst, folder),
        ))
    return moves


def migrate(folder, dry_run=False):
    moves = plan_moves(folder)
    if not moves:
        print("Nothing to migrate (already migrated, or no runs found).")
        return

    runs = sorted({m[3].split(os.sep)[1] for m in moves})
    print("{} runs, {} files to move.".format(len(runs), len(moves)))
    for src, dst, src_rel, dst_rel in moves:
        print("  {:52s} -> {}".format(src_rel, dst_rel))
        if not dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

    if dry_run:
        print("\n(dry run -- nothing moved)")
        return

    manifest = {"moves": [[m[2], m[3]] for m in moves]}
    with open(os.path.join(folder, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nMigrated {} files into {}/runs/. Manifest: {}".format(
        len(moves), os.path.relpath(folder), MANIFEST))


def reverse(folder, dry_run=False):
    manifest_path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(manifest_path):
        print("No {} found -- nothing to reverse.".format(MANIFEST))
        return
    with open(manifest_path) as f:
        moves = json.load(f)["moves"]

    print("Reversing {} moves.".format(len(moves)))
    for src_rel, dst_rel in moves:
        src = os.path.join(folder, src_rel)   # original location
        dst = os.path.join(folder, dst_rel)   # current (nested) location
        if not os.path.isfile(dst):
            print("  !! missing, skip: {}".format(dst_rel))
            continue
        print("  {} -> {}".format(dst_rel, src_rel))
        if not dry_run:
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)

    if dry_run:
        print("\n(dry run -- nothing moved)")
        return

    # Prune now-empty run dirs and the manifest.
    runs_root = os.path.join(folder, "runs")
    for dirpath, _, _ in os.walk(runs_root, topdown=False):
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
    os.remove(manifest_path)
    print("\nReversed. Removed manifest and empty run dirs.")


def main():
    ap = argparse.ArgumentParser(description="Migrate a flat model folder to per-run dirs.")
    ap.add_argument("--folder", required=True, help="model folder, e.g. models/h2o-h2o")
    ap.add_argument("--reverse", action="store_true", help="undo a previous migration")
    ap.add_argument("--dry-run", action="store_true", help="show moves without doing them")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        ap.error("folder not found: {}".format(folder))

    if args.reverse:
        reverse(folder, dry_run=args.dry_run)
    else:
        migrate(folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
