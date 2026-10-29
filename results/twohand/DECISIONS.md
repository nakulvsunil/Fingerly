# Part C: decisions where the README is ambiguous or silent

Each entry says what the README leaves open, what we chose (the simplest option), and where it lives.

## Phase 1: data, actions, reward

1. **Part A code on this branch.** `two-hand-monophonic` was cut from `one-hand`, which has no Part A
   code. The Part A files (`fingerly/paper/`, `fingerly/metrics.py`, `reproduce.py`,
   `tests/test_paper.py`, `data/paper/`) were copied from `single-hand-monophonic` without a merge,
   so the README stays as edited. Part A's run outputs (`runs/paper/`) are not copied.
2. **Tolerances.** Defaults kept: `chord_tol` 0.03 s, `overlap_tol` 0.05 s. On training pieces only
   a handful of within-hand onset differences fall below 30 ms, and the overlap histogram has a gap
   at 50–120 ms between legato overlaps and held notes (`coverage.md`, `overlap_hist.png`).
3. **Overlap rule.** Each note is compared with the first later onset group of its hand, i.e. the
   first note more than `chord_tol` later. The original offset is kept for held notes
   (`hold_until`) and for rest detection.
4. **Same key twice in one chord** (two annotated notes on one key at once): one is kept.
5. **Segment minimum length.** `--min-len 16` applies to training segments. Validation and test
   keep every segment of at least 2 events, so the metrics cover almost every note of the 30 test
   pieces. Dropping test segments shorter than 16 events would leave notes unscored.
6. **Splits.** Test = the 30 PIG IDs of §C.4.6. They match the `in_30_subset` column of
   `eer2025_table_a1.csv`. Validation = 12 of the remaining 120 pieces, shuffled with seed 0;
   training = the other 108.
7. **Mirrored space.** Both cost models negate key values for the left hand (semitones for
   `parncutt12`, invisible-black-key positions for `balliauw15`) and apply the right-hand rules.
   For `balliauw15` this equals the README's X_L(f, g) = X_R(g, f) reading (mirror check: 0 of
   30,625 cases differ). Rules that compare pitch order (`balliauw15` 7b and 9, `parncutt12`
   rule 4 "between", rule 12 thumb passing) therefore use mirrored order for the left hand.
8. **Chord pairs between events.** `all` = every (previous note, current note) pair.
   `extremes` = lowest→lowest and highest→highest only (the outer voices). Both readings are
   implemented; `all` is the default.
9. **Rules 4–12 (`parncutt12`) and the mono rules of `balliauw15` with chords nearby.** A rule
   is charged only when every event it looks at is a single note. A previous- or next-note colour
   term (thumb or five on black) is skipped when that neighbour is a chord.
10. **`balliauw15` rule 7(b).** Implemented as printed: n_{t−2} < n_{t−1} < n_t, the middle note
    played by the thumb, and (t−2, t) outside [MinPrac, MaxPrac]. `--rule7b any-thumb` (thumb on
    any of the three) is also available. It changes no Table A1 count.
11. **Same finger on a different key.** `parncutt12` keeps Part A's `zero-span`. `balliauw15` uses
    the diagonal of Eq. 8 as printed (0), so rules 1–3 charge a same-finger move per unit,
    including rule 3's 10 points per unit.
12. **Feasibility masks.** Chord reach checks adjacent chord notes against [MinPrac, MaxPrac] of
    the cost model. The single-note → single-note transition is masked by the same limits. The
    same finger always passes; its cost comes from the rules. Chord transitions are not masked.
13. **Retroactive charges.** The parts of rules that look at the hand's next note (`parncutt12` 10
    and 11, `balliauw15` 14 and 15) are separated out as `next_terms`. Immediate and deferred
    charging give identical totals on every PIG stream checked and in the unit tests.
14. **Hand position for chords.** The thumb-side note is the lowest note (right hand) or the
    highest (left hand), as for the features in §C.7.
15. **Human fingerings in the checks** are built from each annotator's own hand assignment and
    fingers. Notes an annotator left unfingered are dropped.
16. **v_max** = 95th percentile of |thumb shift| / Δt over all consecutive same-hand events of every
    annotator on the training pieces, including events with no shift: 38.9 semitones/s. Excluding
    zero-shift transitions would give 44.1. The README's wording ("human hand speed") does not
    say which; the first is used.
