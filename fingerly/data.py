"""One place that turns command-line options into passages, so train/evaluate/predict agree."""

from __future__ import annotations

from typing import Optional

from .pig import Split, coverage, load_pig, make_split
from .synth import synthetic_passages


def build_split(
    pig_dir: Optional[str],
    hand: str = "R",
    min_len: int = 8,
    val_pieces: int = 12,
    split_seed: int = 0,
    verbose: bool = True,
) -> Split:
    if pig_dir is None:
        raise ValueError("--pig-dir is required (or use --synthetic-only for a smoke test)")
    pig = load_pig(pig_dir)
    split = make_split(pig, hand=hand, min_len=min_len, val_pieces=val_pieces, seed=split_seed)
    if verbose:
        kept, total = coverage(pig, hand=hand, min_len=min_len)
        print(f"PIG: {len(pig)} pieces; {hand}-hand notes kept in monophonic passages: "
              f"{kept}/{total} ({100 * kept / max(total, 1):.1f}%)")
        print(split.summary())
    return split


def synthetic_split(n_train: int, hand: str = "R", seed: int = 0) -> Split:
    """Label-free split for smoke tests without PIG."""
    train = synthetic_passages(n_train, hand, seed)
    val = synthetic_passages(max(20, n_train // 10), hand, seed + 1)
    return Split(train=train, val=val, test=val, train_pieces=[], val_pieces=[], test_pieces=[])
