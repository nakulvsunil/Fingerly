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

## Phase 2: state, environments, cost tables, search, speed

17. **Mode 2a** (`--mode lh`) is the left hand's events of each segment, played alone: Part B
    covered the right hand, and the README names the flag `lh`.
18. **"A hand whose key is still sounding"** (2c hand mask) uses the note's offset *after* the
    overlap processing of §C.4.2. Held notes are clipped at the hand's next onset, so a hand
    holding a bass note can still play. With the original offsets, a hand holding a long note
    could not play until the note ended, and the other hand had to take every one of its notes.
    On validation that pushed 2c's optimum far above 2b's (411 vs 109 on one segment).
19. **M0 ignores occupied fingers.** They depend on choices further back than the DP/beam state
    holds. M0 is therefore a lower bound on what the environment allows; the gap to it is never
    negative in 2a/2b.
20. **Forced fallbacks.** Shift (no set reachable from the hand's fingers): all reachable sets of
    the right size, cost 2. Unreachable chord (no set passes the reach mask): all sets of the size,
    also cost 2, logged as forced. Occupied fingers that block every set are released without a
    cost, logged as forced. All three count as infeasible transitions in the metrics.
21. **Features: 67 numbers with K = 4**
    (6 current + 7 per look-ahead event + 16 per hand + 1 mode; `features.py` lists them).
    "Key" is the thumb-side note: right lowest, left highest. In 2c the current event's register
    uses its lowest note. Time features are clipped at 2 s and scaled to [0, 1].
22. **`--no-mask`** removes the feasibility masks (reach, transition, occupied, 2c hand masks)
    and charges 10 per infeasible choice. The chord-size mask stays: a set of the wrong size has
    no meaning. Chord events also keep their annotated hand.
23. **Step counting.** Steps are environment steps: one vector step of 16 environments = 16
    steps. "Update every 4 environment steps" = 4 updates of batch 256 per vector step.
24. **GPU updates use a CUDA graph.** On the GPU the whole update (forward, Double DQN target,
    Huber loss, optional DQfD margin, gradient clipping, Adam, soft target update) is captured
    once and replayed: 0.6 ms instead of 3.6 ms. The environments always run on the CPU.
25. **"Brute force"** for the beam check = the same search with no pruning. States are merged
    only when their futures are identical, so this is exact. It is checked against plain
    enumeration of every action sequence on segments of 4 events (unit tests).

## Phase 3: methods

26. **Synthetic exercises** (§C.4.7): 200 two-hand segments are added to every training pool
    (`--synthetic 200`): contrary-motion scales, Alberti bass under a melody, arpeggios over 4
    octaves split between the hands, melodies passed between the hands, block chords. They get no
    imitation bonus and are not used in any metric.
27. **2c curriculum:** short synthetic segments (≤ 32 events) for the first 10% of steps, all
    synthetic exercises until 25%, then PIG plus synthetic.
28. **S1** samples come from playing every annotator's fingering of every training segment
    through the environment. An event whose annotated choice is unavailable (other hand, a chord
    not in pitch order, or masked) gets the first allowed action and is not used as a sample:
    2,496 of 53,839 events. S1 is the agent's network (dueling 256-256) with cross-entropy on the
    masked logits: Adam, lr 1e-3, batch 256, 20 epochs, best validation Mgen.
29. **S2** is deterministic (an HMM fit by counting), so it is run once, not with 3 seeds. Its
    emission is the mirrored interval between consecutive thumb-side notes of a hand; the
    transition is renormalised over the sets of the next event's size. Like Part B's HMM, it is
    unmasked: its infeasible moves are counted, not prevented.
30. **A2 demonstrations** play annotator 1's fingering of each training segment and stop (with a
    bootstrapped n-step target) at the first annotated choice that is unavailable.
31. **A3** seeds the elite memory with M0's optimal episode for every training segment (exact DP
    in 2b). These seeded entries are never replaced.
32. **2c initialisation:** only the shared trunk (`body.*`) is copied from the 2b checkpoint. The
    heads have 31 vs 62 outputs and start fresh.
33. **R4 and tabular Q-learning use the test pieces' scores** (rule reward only, no labels, no
    bonus), because the README defines them per test piece (§C.9.2, §C.9.5). Every other method
    sees no test piece. In R4 and Q-learning, one episode is one pass through all of the piece's
    segments.
34. **Benchmarks** (DQN, MaskablePPO) use λ = 0.5, the README's default example. Round 1 runs before
    the best λ is known. Q-learning uses the rule reward only.
35. **Model selection:** best validation Mgen when λ > 0 (R2–R4, A1–A3, 2c, DQN, PPO); lowest
    validation difficulty per note when λ = 0 (R1, the λ = 0 ablation).

## Phase 4: evaluation

36. **Human difficulty** (the d_rel denominator) = the mean over a piece's annotators of the rule
    points of each annotator's own fingering of the same segments, under each annotator's own
    hands. Chords an annotator split across hands are split. The 2025 paper uses the
    lowest-difficulty annotator for Table A1; the lowest is also stored (`difficulty_min`).
    d_rel = Σ agent ÷ Σ human over the pieces; the mean of per-piece ratios is also reported.
37. **Success rate** = segments with no infeasible or forced transition and gap ≤ 10%, pooled
    over seeds, with a Wilson interval. In 2c, M0 is the beam search, so a few regrets can be
    negative; the count is reported.
38. **AUC** of the validation return curve up to 1M steps, rescaled so a random-valid policy is
    0 and M0 is 1 on validation, clipped to [0, 1].
39. **"Very hard" transition** (sequence breakdowns) = rule points at the event above the 95th
    percentile of annotator 1's per-event points on the same split.
40. **Crossing rate** = events after which the left thumb is above the right thumb (both hands
    placed). **Hand-switch rate** = consecutive single-note events played by different hands.
    **Large-stretch rate** uses the evaluated method's own cost model.
41. **Composer eras** come from PIG's List.csv: Baroque (Bach, Scarlatti); Classical (Mozart,
    Beethoven); Romantic (Chopin, Schumann, Schubert, Mendelssohn, Liszt, Brahms, Grieg,
    Tchaikovsky, Dvořák, Saint-Saëns, Mussorgsky); late Romantic / 20th century (Debussy, Ravel,
    Rachmaninoff, Scriabin, Satie, Fauré, Joplin, Bartók, Albéniz).
42. **Robustness:** tempo, transposition and onset jitter change the raw PIG notes; events, chords,
    held notes and segments are then rebuilt. Each note gets the same change for every annotator,
    so the labels still match. The stress set has 40 synthetic segments (20 with 2–3-octave leaps,
    20 with hands alternating every 80 ms); it has no labels, so only difficulty, infeasible rate
    and SR are reported for it.
43. **Energy** is an estimate: training hours × 45 W (the i7-12650H's base power). No power meter
    is read.

## Phase 5: experiment runner

44. **Choosing λ** (after round 1) and the best R configuration (after round 2) uses validation
    only: the distance of each configuration's validation result to the target region,
    max(0, d_rel − 0.81) + max(0, 0.62 − Mgen), ties broken by the higher Mgen. The best two λ go to
    R3 (E4).
45. **E5 accelerators** use the R3 configuration at the best λ (A3 needs the elite memory, and R3
    is the full method).
46. **E6 R4** fine-tunes each seed's checkpoint of the best R configuration, so 3 runs of 30
    pieces × 50 episodes. **E7 2c** initialises each seed from the same seed's 2b checkpoint.
47. **E10, the "− crossing term" ablation,** is run in 2c, the mode where the crossing term exists,
    and compared with 2c. The other ablations are compared with the full method (R3, best λ, 2b).
48. **E12** runs every listed value, including the defaults, so each setting's plot has all its
    points (19 runs of 500k steps, 1 seed).
49. **Concurrency:** 6 jobs at once, at most 4 on the GPU; a job starts only when at least 2.5 GB of
    RAM is free.
50. **Early stopping** (patience 50k steps, validation every 25k, §C.9.4) is only allowed after the
    ε-decay phase (the first 20% of steps), so no run stops while still exploring. With
    evaluations every 25k, two evaluations without improvement end a run. The first R1 run stopped
    at 275k of 1M steps, while its validation return was still rising slowly. This is the README's
    setting and is kept; the final report says when it may have cut learning short.
