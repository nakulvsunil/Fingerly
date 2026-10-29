# Fingerly — reproducing Iman & Ahn (2024), then extending it

Model-free deep RL for piano fingering. The repository has two parts:

- **Part A: paper reproduction.** A faithful rebuild of A. P. Iman and C. W. Ahn, *"A Model-Free
  Deep Reinforcement Learning Approach to Piano Fingering Generation"*, IEEE CAI 2024, pp. 31–37,
  DOI 10.1109/CAI59869.2024.00016. It covers the same four right-hand monophonic pieces, the same
  state, the same 12-rule reward, the same four algorithms (DQN, DDQN, PG, A2C), the same
  hyperparameters and the same tables. This is Stage 0 of the proposal ("Reproduce").
- **Part B: the Fingerly stage-1 setup.** A masked DQN / Double DQN on the whole PIG dataset, with
  a train/val/test split and a simplified cost table. It goes beyond the paper and is the base for
  the two-hand stages.

> **Status.** Part A's code (`fingerly/paper/`, `reproduce.py`) is implemented; see the checklist
> in §A.8 for what is still open (FHMM1–3, and checking the V0 transcription against the score).
> Results are written to `results/paper/tables.md`. Part B's code is unchanged.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # a few minutes; includes a short DQN training run
```

Get PIG from <http://beam.kisarazu.ac.jp/research/PianoFingeringDataset/>. It requires the usage
agreement, so do not commit it. Unzip it, for example to `data/PianoFingeringDataset_v1.2`.
`--pig-dir` can point at the root or straight at `FingeringFiles/`.

No dataset yet? `python -m fingerly.fake_pig --out data/fake_pig` writes a fake dataset in PIG
format so you can test the pipeline. The numbers it produces are meaningless.

---

# Part A — Reproducing Iman & Ahn (2024)

## A.1 What the paper did

| Item | Paper (Sec. III–IV) |
| --- | --- |
| Task | Right hand only, monophonic, one finger (1–5) per note |
| Environments | `PianoFingering-V0`…`V3`, one piece each (§A.2) |
| Episode | One pass through the piece, note by note |
| State | s_t = (n_{t−2}, a_{t−2}, n_{t−1}, a_{t−1}, n_t): last two notes and their fingers, plus the current note (MIDI numbers) |
| Encoding | One-hot: each note over 88 keys (A0–C8), each finger over 5, concatenated. The paper says 1×273 (see §A.7, item 1) |
| Action | Finger 1–5. **No action mask**: bad moves are only penalised by the reward |
| Reward | R_{t+1} = −Σ_{k=1}^{12} d_k(s_t, a_t): Parncutt et al. (1997) difficulty rules on the paper's Table I (§A.3) |
| Algorithms | DQN, Double DQN (off-policy); REINFORCE with a state-value baseline ("PG") and A2C (on-policy) |
| Networks | MLP 512 → 256 → 64, ReLU. Q head: 5 outputs. V / critic head: 1 output. Policy head: 5 outputs with softmax |
| Training | 500 episodes, γ = 0.99, Adam with lr = 0.001, MSE loss |
| DQN / DDQN | Replay of the latest 10,000 steps, batch 32, target network synced every 100 steps |
| Runs | 10 per algorithm per environment |
| Metrics | Difficulty level L = D / N (mean difficulty per note); match rates Mgen, Mhigh, Msoft, Mrec (Nakamura et al., 2020) |
| Hardware | One RTX 2080 (8 GB) |

## A.2 The four pieces

| Env | Piece | Source | What to do |
| --- | --- | --- | --- |
| V0 | Beyer, *Vorschule im Klavierspiel* Op. 101 No. 12 | **Not in PIG** | Transcribe the right hand with the composer's fingering into PIG format at `data/paper/V0_beyer_op101_no12_fingering.txt`. Beyer (1850) is public domain, so this file can be committed |
| V1 | Chopin, Waltz Op. 64 No. 1 | PIG | Find the ID (below) |
| V2 | Beethoven, *Für Elise* | PIG | Find the ID (below) |
| V3 | Scarlatti, Sonata in A major K. 208 | PIG | Find the ID (below) |

The paper does not give the PIG file IDs. Look the three pieces up in the piece list that ships with
PIG, and record them in `data/paper/pieces.json`:

```bash
python reproduce.py pieces --pig-dir data/PianoFingeringDataset_v1.2   # searches titles, writes data/paper/pieces.json
```

Keep only the right-hand notes (positive finger numbers in PIG; take the first finger of
substitutions such as `3_1`). The paper picked these pieces because their right hand is
monophonic. `pieces` must report any right-hand chord or overlap it finds rather than silently
dropping it. List the annotators available for each piece: match rates need several of them.

V0 sanity numbers from the paper: the best episode reward is **−18**, and the best L is **0.6207**,
which equals 18 / 29. That implies 29 scored notes. Fig. 5 shows the composer's fingering as
`1234 5432 1234 5432 1234 5432 3132 1`, which is also 29 notes.

## A.3 Reward: the 12 rules

**Distance convention.** For a finger pair (i, j) with i < j, the distance is
d = pitch(finger j) − pitch(finger i), in semitones. It is negative when the fingers are crossed,
for example thumb above finger 3. Rules 1–3 apply to consecutive notes (t−1, t). Rules 4–5 apply to
notes (t−2, t) with fingers (a_{t−2}, a_t).

**Table I of the paper (right hand, semitones), as printed:**

| Pair | MinPrac | MinComf | MinRel | MaxRel | MaxComf | MaxPrac |
| --- | --- | --- | --- | --- | --- | --- |
| 1-2 | −5 | −3 | 1 | 5 | 8 | 10 |
| 1-3 | −4 | −2 | 3 | 7 | 10 | 12 |
| 1-4 | −3 | −1 | 5 | 9 | 12 | 14 |
| 1-5 | −1 | 1 | 7 | 10 | 13 | 15 |
| 2-3 | 1 | 1 | 1 | 2 | 3 | 5 |
| 2-4 | 1 | 1 | 3 | 4 | 5 | 7 |
| 2-5 | 2 | 2 | 5 | 6 | 8 | 10 |
| 3-4 | 1 | 1 | 1 | 2 | 2 | 4 |
| 3-5 | 1 | 1 | 3 | 4 | 5 | 7 |
| 4-5 | 1 | 1 | 1 | 4 | 3 | 5 |

**Rules, as the paper states them:**

1. **Stretch:** 2 points per semitone below MinComf or above MaxComf.
2. **Small span:** per semitone below MinRel, 1 point for pairs with the thumb and 2 points for
   other pairs.
3. **Large span:** per semitone above MaxRel, 1 point for pairs with the thumb and 2 points for
   other pairs.
4. **Position-change count:** 1 point if the distance between notes t−2 and t is outside
   [MinComf, MaxComf]. Add 1 more point if note t−1 lies between them, is played by the thumb, and
   that distance is outside [MinPrac, MaxPrac].
5. **Position-change size:** 1 point per semitone that the distance between notes t−2 and t falls
   outside [MinComf, MaxComf].
6. **Weak finger:** 1 point when finger 4 or 5 is used.
7. **Three-four-five:** 1 point when fingers 3, 4 and 5 occur consecutively, in any order.
8. **Three-to-four:** 1 point when finger 4 comes right after finger 3.
9. **Four-on-black:** 1 point when fingers 3 and 4 are consecutive (either order), with 3 on a
   white key and 4 on a black key.
10. **Thumb-on-black:** 1 point for the thumb on a black key, +2 if the next note is white, and +2
    if the previous note is white.
11. **Five-on-black:** 2 points for finger 5 on a black key when the previous note is white, and +2
    more if the next note is white.
12. **Thumb passing:** 1 point when the thumb passes on the same level (white to white, or black to
    black). Add 3 more points if the lower note is white and played by a non-thumb finger while the
    upper note is black and played by the thumb.

At the start of a piece, rules that need notes that do not exist yet are skipped. On note 1, only
rules 6, 10 and 11 can apply (without their "previous note" parts). Rules 4, 5 and 7 start at note 3.

## A.4 Check the reward before training anything

Wrong rules show up as wrong numbers long before the agent learns anything, so `check-reward` runs
three tests with no training at all:

```bash
python reproduce.py check-reward --pig-dir data/PianoFingeringDataset_v1.2
```

1. **V0, composer's fingering: total must be exactly 18.** If the passage is a stepwise
   five-finger pattern on white keys, as Fig. 5 shows, the only rules that fire are:
   - Rule 6: 9 points (one 4 in each `1234`, and a 5 and a 4 in each `5432`).
   - Rule 7: 6 points (3-4-5 and 5-4-3 in each of the three cycles).
   - Rule 8: 3 points (3→4, three times).

   This also pins down rule 7's reading: it counts triples containing 3, 4 *and* 5. A reading that
   also counts 4-5-4 gives 21.
2. **The "Human" row of Table II.** Scoring the PIG annotation should give L = 1.8419 (V1),
   4.0441 (V2) and 2.9155 (V3). The paper does not say which annotator it used, so score every
   annotator and report which one matches.
3. **Exact optimum by dynamic programming.** Every rule depends only on (a_{t−2}, a_{t−1}, a_t) and
   on fixed pitches. A DP over the 25 pairs (a_{t−1}, a_t) therefore finds the exact
   minimum-difficulty fingering. On V0 it should not go below 18, because the paper reports −18 as
   the maximum reward. If it does, your same-finger handling (§A.7, item 2) is more lenient than the
   authors'. The DP also gives every piece a floor to compare the agents against.

## A.5 Train

The defaults are the paper's settings. Only settings the paper leaves open have our own defaults
(§A.7).

```bash
# one cell of the grid: DQN on V0, 10 runs
python reproduce.py train --env V0 --algo dqn --runs 10 --out runs/paper

# the full grid: {V0,V1,V2,V3} × {dqn,ddqn,pg,a2c} × 10 runs
python reproduce.py train --env all --algo all --runs 10 --out runs/paper
```

Switches and their defaults: `--episodes 500 --gamma 0.99 --lr 1e-3 --buffer 10000 --batch 32
--target-sync 100 --eps-start 1.0 --eps-end 0.01 --eps-decay 0.999 --same-finger zero-span
--table1 paper --seed-base 0`.

Each run saves `runs/paper/<env>/<algo>/run<k>/` containing the per-episode reward, the final
weights, the greedy fingering, and the wall-clock time.

**Budget.** By the paper's own timings (Tables III–IV), one run of each algorithm on each piece takes
about 45 minutes on an RTX 2080. The full 10-run grid therefore takes about 7.5 GPU-hours. V0 alone
takes about 27 minutes for all algorithms × 10 runs, which matches the paper's figure.

## A.6 Evaluate and compare with the paper

```bash
python reproduce.py report --runs runs/paper --pig-dir data/PianoFingeringDataset_v1.2 --out results/paper
```

This writes `tables.md`, which reproduces Tables II–V, and `fig4_v0_curves.png`, which shows the mean
± std episode reward on V0 per algorithm (Fig. 4).

**Protocol.**
- After training, each run is rolled out greedily (ε = 0) once.
- **Table II** reports the *least* difficult L over the 10 runs.
- **Tables III–IV** report the mean ± std of L over the 10 runs, plus the mean run time.
- **Table V** compares the greedy fingerings with the PIG annotations of V1–V3, averaged over
  pieces. Per-piece values are written too.
- The FHMM1–3 rows come from Nakamura et al.'s HMM programs, run on the same passages and scored
  with our rules for Table II.

**Targets from the paper.**

Table II — best L over 10 runs (lower is better):

| Method | V0 | V1 | V2 | V3 |
| --- | --- | --- | --- | --- |
| DQN | 0.6207 | 1.3721 | 3.1814 | 2.5441 |
| DDQN | 0.6207 | 2.5163 | 5.2647 | 3.25 |
| PG | 8.1034 | 10.712 | 14.478 | 11.294 |
| A2C | 2.7586 | 3.8930 | 7.0441 | 6.6176 |
| FHMM1 | 1.0697 | 2.3349 | 3.6029 | 2.8320 |
| FHMM2 | 0.6207 | 1.8884 | 3.3235 | 2.8028 |
| FHMM3 | 0.6207 | 2.2512 | 3.2941 | 2.9859 |
| Human | 0.6207 | 1.8419 | 4.0441 | 2.9155 |

Tables III–IV — mean ± std of L over 10 runs:

| Method | V0 | V1 | V2 | V3 |
| --- | --- | --- | --- | --- |
| DQN | 1.2552 ± 0.905 | 1.51 ± 0.17 | 3.78 ± 0.30 | 3.14 ± 0.53 |
| DDQN | 0.7000 ± 0.238 | 2.92 ± 0.21 | 6.05 ± 0.78 | 3.76 ± 0.35 |
| PG | 9.7138 ± 1.267 | 11.61 ± 0.39 | 16.5 ± 0.98 | 13.38 ± 1.05 |
| A2C | 3.7448 ± 2.759 | 5.08 ± 0.83 | 8.94 ± 1.33 | 7.75 ± 0.56 |

Table V — match rates (higher is better):

| Method | Mgen | Mhigh | Msoft | Mrec |
| --- | --- | --- | --- | --- |
| DQN | 66.50% | 68.97% | 78.52% | 72.55% |
| DDQN | 58.23% | 61.10% | 70.27% | 62.29% |
| PG | 24.20% | 26.97% | 34.60% | 28.16% |
| A2C | 43.08% | 46.30% | 54.18% | 47.02% |
| FHMM1 | 63.50% | 66.11% | 79.47% | 74.22% |
| FHMM2 | 61.32% | 63.96% | 76.61% | 70.41% |
| FHMM3 | 60.62% | 61.34% | 75.42% | 70.17% |

**What counts as reproduced.** Exact numbers will differ because of seeds and the gaps listed in
§A.7. The claims to reproduce are:
- DQN and DDQN reach the V0 optimum (−18; the paper reports this at episode 124 for DQN and episode
  202 for DDQN).
- The ordering in L is DQN ≤ DDQN < A2C < PG on every piece.
- DQN's Mgen is close to or above FHMM1's. The proposal's exit benchmark is DQN within about
  2 points of the HMM's Mgen on these pieces.

Report any claim that does not hold, together with which §A.7 choice might explain it.

**Metric definitions.**
- **Mgen:** match rate averaged over annotators.
- **Mhigh:** match rate against the best-matching annotator.
- **Msoft:** a note counts as matched if any annotator agrees with it.
- **Mrec:** Nakamura et al.'s recombination match rate, computed against the best fingering
  stitched together from the annotators' sequences. Implement its switching rule from Nakamura,
  Saito & Yoshii (2020).

For pieces with a single annotator, only Mgen is defined.

## A.7 Gaps and inconsistencies in the paper, and the defaults we chose

Each choice below has a flag or a note in `reproduce.py`. Say which you used in the report.

1. **Input size.** The paper describes 3 notes × 88 + 2 fingers × 5 = 274 inputs but writes
   1×273. We use 274.
2. **Same finger on a different key.** Table I has no (i, i) pair, and the paper is silent on this
   case. If such moves cost nothing, playing every note with finger 2 scores 0, which beats the
   reported optimum of 18. So the authors must have charged for it. Our default,
   `--same-finger zero-span`, sets all six limits to 0 for (i, i). Rules 1–3 then charge per
   semitone moved. Use the V0 DP check (§A.4) to confirm that the optimum stays at 18.
3. **Table I, row 4-5.** MaxRel (4) is larger than MaxComf (3), which looks like a typo. Default:
   `--table1 paper`, as printed. Check against Parncutt et al. (1997), Table 1, and add
   `--table1 parncutt` if it differs, then report both.
4. **Exploration** is not described. Our default is ε-greedy, decayed per step from 1.0 by a factor
   of 0.999, with a floor of 0.01. That puts ε at about 0.03 near episode 120 on V0, which makes a
   −18 training episode around episode 124 plausible. Tune and report.
5. **The start of a piece.** Missing n_{t−2}, a_{t−2}, n_{t−1} and a_{t−1} are encoded as all-zero
   one-hot blocks.
6. **Next note in the reward, but not in the state.** Rules 10 and 11 look at the next note, which
   the state does not contain. We keep this as written, so the agent sees a partially observed
   reward.
7. **Thumb passing (rule 12).** A "pass" is read as a crossing: the thumb plays one of two
   consecutive notes, and the finger order is inverted relative to the pitch order.
8. **On-policy update details.** PG updates once per episode on Monte-Carlo returns minus V(s).
   A2C updates every step with a one-step TD advantage. Both use the same learning rate, and the
   critics use MSE.
9. **Final or best policy.** Table II uses each run's final greedy policy. `report` also logs the
   best training episode, in case that is what the authors meant.
10. **Annotators and aggregation** for the "Human" row and for Table V are not stated. Score every
    annotator, average Table V over V1–V3, and report the per-piece values as well.
11. **Run time** depends on hardware. Compare times only as ratios between algorithms.

## A.8 Implementation checklist

- [x] `fingerly/paper/rules.py`: Table I, the 12 rules returning a per-rule breakdown, and the
  `--same-finger` and `--table1` options
- [x] `fingerly/paper/pieces.py`: V0 loader, PIG ID lookup, right-hand stream, chord and overlap
  report
- [x] `data/paper/V0_beyer_op101_no12_fingering.txt`: the transcription; passes the 18-point test.
  **Reconstructed** from the Fig. 5 fingering assuming the hand stays in C position; still to be
  checked against the printed score
- [x] `fingerly/paper/env.py`: Gymnasium environment with the paper's state, 274-dimensional
  one-hot encoding, no mask, and one pass per episode
- [x] `fingerly/paper/dp.py`: exact second-order DP (25 pair states)
- [x] `fingerly/paper/agents.py`: DQN, DDQN, REINFORCE with baseline, A2C, using the 512-256-64 MLP
- [x] `fingerly/metrics.py`: add Mrec (our reading of the switching rule; see `recombination_match`)
- [x] `reproduce.py` with the subcommands `pieces`, `check-reward`, `train` and `report`
- [ ] FHMM1–3 outputs for V1–V3 under `data/paper/fhmm/` (Nakamura et al.'s programs are not in
  the repository; `report` shows the paper's printed FHMM numbers instead)
- [x] Tests (`tests/test_paper.py`): the V0 total is 18, the rule-7 reading, and the DP equals the
  brute-force optimum on short random passages

**Findings so far** (`python reproduce.py check-reward`):
- PIG IDs: V1 = 025 (5 annotators), V2 = 061 (2), V3 = 140 (2). The paper's L × N gives whole
  numbers for N = 215 / 136 / 68, the full right-hand streams, so we use those (0 chords; 52, 2 and
  0 legato overlaps).
- The V0 DP optimum is exactly 18 with `--same-finger zero-span`; with `free` it is 0, as predicted.
- The Human row does not match exactly: annotator 1 scores D = 421 / 562 / 213 against the paper's
  396 / 550 / ≈198 (L 1.958 / 4.132 / 3.132 vs 1.8419 / 4.0441 / 2.9155). No combination of the
  §A.7 readings (same-finger handling, rule 11's "further" points, clipping rules 2–3 at the
  comfortable range) matches all three, so at least one rule is read differently by the authors.

---

# Part B — Fingerly stage-1 setup (beyond the paper)

This part fingers **one hand at a time** (right hand by default, `--hand L` for the left) on
**monophonic passages** from the whole of PIG. A masked DQN / Double DQN agent assigns a finger to
every note using a simplified ergonomic reward. Hand choice and the 10-action space come in the next
stage.

### How Part B differs from the paper

| | Paper (Part A) | Part B |
| --- | --- | --- |
| Data | 4 single pieces; trains and evaluates on the same piece | All monophonic PIG passages; split into train, validation and test (the 30 multi-annotator pieces) |
| State | Last two notes and fingers plus the current note, one-hot (274) | 24 features: relative intervals, a 3-note look-ahead, the last two fingers |
| Reward | 12 Parncutt rules | Simplified cost table (below), with an optional expert bonus λ |
| Actions | Unmasked | Masked to feasible fingers (`--no-mask` for the ablation) |
| Agent | Plain DQN / DDQN, PG, A2C | Double DQN with a dueling head and n-step returns (`--no-double --no-dueling` for plain DQN) |
| Budget | 500 episodes | Steps (`--steps`, 200k by default) |

### Run

```bash
# train (≈ 5–10 min on a laptop CPU for 200k steps)
python train.py --pig-dir data/PianoFingeringDataset_v1.2 --hand R --seed 0 --out runs/rh_ddqn_s0

# evaluate agent(s) + all baselines on the 30 multi-annotator test pieces
python evaluate.py --pig-dir data/PianoFingeringDataset_v1.2 --ckpt "runs/rh_ddqn_s*/best.pt" --show 2 --out results/rh.json

# learning curves
python plot.py runs/rh_ddqn_s0 --out results/curves.png

# finger a PIG-format file
python predict.py --ckpt runs/rh_ddqn_s0/best.pt --input .../FingeringFiles/001-1_fingering.txt --output out/001.txt

# watch the agent play it on the pygame piano (visual/piano)
python visualize.py --ckpt runs/rh_ddqn_s0/best.pt --input .../FingeringFiles/001-1_fingering.txt
python visualize.py --policy dp --input .../001-1_fingering.txt --passage 0 --speed 0.75   # or greedy / human

# everything, 3 seeds
bash run_seeds.sh data/PianoFingeringDataset_v1.2
```

Useful switches:
- `--no-double --no-dueling`: plain DQN, as in Iman & Ahn
- `--lam 0.5`: expert-match bonus
- `--synthetic 500`: add scales, arpeggios and melodies
- `--no-mask`: ablation in which the agent can make impossible moves and pays the penalty
- `--span-scale 0.85`: a smaller hand
- `--select-by match`: keep the checkpoint with the best validation Mgen instead of the best
  validation reward
- `--steps`, `--lr`, `--n-step`

### Design

**Data.** For each piece, the note list and the hand of each note come from annotator 1. A hand's
notes form a passage as long as each note starts after every earlier note of that hand has ended.
Chords, and notes played under a held note, end the passage and are dropped. Passages need at least
8 notes (`--min-len`). The test set is the pieces fingered by several pianists (30 in PIG). Of the
rest, 12 pieces (seeded) are validation and the others are training. Pieces are never split.
`train.py` prints how many notes the filter keeps.

**State** (24 numbers with a look-ahead of 3):
- The current note: distance from middle C / 24, and whether it is black.
- The next 3 notes: interval / 12, whether each is black, and whether it exists.
- The hand: played yet?, interval from its last key, last finger (one-hot), interval from its
  second-to-last key, second-to-last finger (one-hot). The second-to-last finger follows Iman & Ahn's
  "last two notes and fingers".

Left-hand intervals are mirrored so both hands share one cost model.

**Action and reward.** The action is a finger, 1–5. The reward is
`−α·C_effort − β·P_impossible + λ·B_expert`, using the proposal's table:

| Event | Cost |
| --- | --- |
| Each semitone wider or narrower than comfortable | 1 |
| Finger 4 or 5 | 1 |
| Thumb on a black key | 1 |
| A crossing | 1 |
| Same finger on a different key | 2 |
| Impossible move | 10 |

Comfortable and practical spans per finger pair are Parncutt's Table 1 (the same values as Part A's
Table I).

Three rules were needed that the proposal does not spell out:
1. **Out of reach → hand shift.** If no other finger can reach the next key from the current one,
   the hand must jump. Every finger is then allowed, and the move costs 2.
2. **Repeated note with a new finger** is always allowed and is scored by the comfort rule.
3. **First note** pays only the per-note costs.

With the mask on, the impossible penalty is never charged.

**Optimal-cost DP.** This reward is first-order, so DP computes the minimum-cost fingering exactly.
That makes DP both the ceiling for reward and the reference the agent is measured against. RL earns
its place once the reward stops being first-order: the imitation term, two hands, look-ahead costs
and polyphony.

**Baselines.**
- Random-valid
- Greedy
- Optimal-cost DP
- `hmm-1st-order`: a simplified stand-in for Nakamura et al.'s HMM, not a reproduction of it

Published PIG numbers include chords and both hands, so they are not directly comparable.

**Metrics.** A note matches an annotator only if the annotator gave it to the same hand and the same
finger. Mgen, Mhigh and Msoft are computed per piece, then averaged over pieces. "Human agreement
(LOO)" scores each annotator against the others and serves as the ceiling. The playability metrics
are effort per note, impossible-move rate, crossing rate, shift rate and weak-finger rate.

---

## Layout

| File | What it does |
| --- | --- |
| `fingerly/pig.py` | PIG reader (pitch → MIDI, finger sign → hand, `3_1` → 3), monophonic filter, train/val/test split |
| `fingerly/cost.py` | Part B cost table with Parncutt span limits, and the action mask |
| `fingerly/env.py` | Part B Gymnasium environment (`action_masks()` is sb3-contrib compatible) |
| `fingerly/dqn.py` | Masked DQN / Double DQN, dueling head, n-step returns, replay buffer, training loop |
| `fingerly/baselines.py` | Random-valid, greedy, optimal-cost DP, first-order HMM |
| `fingerly/metrics.py` | Mgen / Mhigh / Msoft (Mrec to be added), human agreement, playability metrics |
| `fingerly/synth.py` | Label-free synthetic exercises |
| `fingerly/fake_pig.py` | Fake PIG-format data for testing |
| `fingerly/export.py` | Fingers a PIG file and exports it, with per-note cost and rules, as JSON for the player |
| `fingerly/paper/` | Part A: `rules.py`, `pieces.py`, `env.py`, `dp.py`, `agents.py` |
| `data/paper/` | V0 transcription, `pieces.json` (FHMM outputs still missing) |
| `tests/test_paper.py` | Part A tests |
| `visual/piano/` | pygame piano, hand drawing, and `fingering_player.py` (plays an exported JSON) |
| `reproduce.py` | Part A command-line tool: `pieces`, `check-reward`, `train` (parallel CPU/GPU workers, resumable), `report` |
| `train.py`, `evaluate.py`, `predict.py`, `plot.py`, `visualize.py` | Part B command-line tools |

## References

- A. P. Iman, C. W. Ahn. "A Model-Free Deep Reinforcement Learning Approach to Piano Fingering
  Generation." *IEEE CAI 2024*, pp. 31–37. DOI 10.1109/CAI59869.2024.00016.
- R. Parncutt, J. A. Sloboda, E. F. Clarke, M. Raekallio, P. Desain. "An Ergonomic Model of
  Keyboard Fingering for Melodic Fragments." *Music Perception* 14(4):341–382, 1997.
- E. Nakamura, Y. Saito, K. Yoshii. "Statistical Learning and Estimation of Piano Fingering."
  *Information Sciences* 517:68–85, 2020.
- V. Mnih et al. "Human-level control through deep reinforcement learning." *Nature* 518:529–533,
  2015.
- H. van Hasselt, A. Guez, D. Silver. "Deep Reinforcement Learning with Double Q-learning." 2015.