# Fingerly stage 2: two-hand monophonic fingering, project plan

This extends Iman & Ahn, *A Model-Free Deep Reinforcement Learning Approach to Piano Fingering
Generation* (IEEE CAI 2024). That paper fingers the **right hand only** and uses **four pieces**.
Stage 1 of this repo (branch `one-hand`) rebuilt their idea with a masked DQN on PIG, one hand
at a time. Stage 2 is the work for the main branch: **both hands at once**, where the agent decides
**which hand and which finger** play each note, and **illegal fingerings cannot be produced**.
Illegal here includes impossible hand crossings.

---

## 1. Problem definition

**Two-hand monophonic.** Each hand plays at most one note at a time, and the two hands play at
the same time (a melody over a bass line, or a two-part invention). Chords inside one hand are
out of scope. Polyphony is stage 3.

We picked this definition by measuring PIG v1.2 (150 pieces, 48,724 notes, annotator 1):

| Definition | Notes kept | Segments | Verdict |
| --- | --- | --- | --- |
| One note at a time across *both* hands | 684 | 52 | too little data |
| **Each hand monophonic, hands concurrent** (segments ≥ 16 notes) | **9,559** | **109** (107 use both hands) | **chosen** |
| At most 2 notes sounding, any hand split | 10,097 | 127 | candidate for the free-split data filter |

5,197 of the kept notes fall in the 69 pieces that several pianists fingered, which become the
test set. The README's figure of 30 such pieces is wrong for v1.2 and gets fixed in M1.

**Input.** A segment is a list of notes in onset order, each with pitch, onset and offset. The
hand split from the annotators is used only to *filter* segments, to *score* results and, in
mode A, as an input.

**Output.** For every note, a hand (L/R) and a finger (1–5).

## 2. MDP formulation

The environment is `TwoHandFingeringEnv` (Gymnasium). One step is one note, in onset order.
Notes with the same onset are ordered low to high, so the left-hand candidate comes first.

### 2.1 Actions

| Mode | Action space | Use |
| --- | --- | --- |
| **B, free split (main)** | 10 = {L, R} × {1..5} | the agent chooses hand and finger; this is the new contribution |
| A, given split | 5 = finger of the hand the score assigns | stepping stone and ablation; same as stage 1 but with a joint state |

### 2.2 State (about 60 numbers, k = 3 notes of look-ahead)

* **Current note:** pitch (relative to middle C) / 24, black key?, duration, time since the previous onset.
* **Next k notes:** interval / 12, black key?, onset gap, exists?
* **For each hand (L and R):**
  * Has it played yet? Is it still holding a note, and for how much longer?
  * Time since its last onset.
  * Interval from its last key, last finger (one-hot).
  * Interval from its second-to-last key, second-to-last finger (one-hot). This is Iman & Ahn's "last two notes and fingers".
  * Thumb-anchor estimate (see H5) relative to the current note.
* **Between the hands:** gap between the two thumb anchors / 12, and whether the current note lies between the hands, left of both, or right of both.

Intervals stay mirrored for the left hand, so both hands share one cost model, as in stage 1.

### 2.3 Hard legality rules (the "shield")

Every rule is enforced by the **action mask**, so an illegal action can never be selected. The
same rules feed the evaluator, which counts violations for unmasked baselines such as HMM, random
and the paper's agents.

| # | Rule | Illegal when |
| --- | --- | --- |
| H1 | Finger span (Parncutt MinPrac/MaxPrac, per finger pair) | the move from the hand's previous finger is beyond its practical span *and* there is no time to shift (see H6) |
| H2 | One note per hand | the hand is still holding a note whose offset is later than this onset |
| H3 | Simultaneous notes | two notes with the same onset are given to the same hand (follows from H2, stated for clarity) |
| H4 | Key collision | both hands would be on the same key at the same time |
| H5 | **Impossible hand crossing** | the hands' footprints would interleave while both are engaged (details below) |
| H6 | Shift time (optional, on in the robustness study) | a jump of *d* semitones within one hand happens faster than *d / v_max* |

**H5 in detail.** Each hand gets a *thumb anchor*: the key its thumb would sit on given the key
and finger just played. For the right hand that is `key − off[f]`, for the left hand `key + off[f]`,
with natural offsets `off = [0, 2, 4, 5, 7]` semitones for fingers 1–5. A hand is **engaged** if it
is holding a note or played within `τ_engage` seconds. Three policies, set by `--crossing`:

* `strict` (default): whenever both hands are engaged, require `anchor_L + gap_min ≤ anchor_R` and
  `key_L < key_R`. No crossing at all.
* `gated`: a cross-over is allowed only if the crossing hand is *free* (holding nothing), the other
  hand's footprint is not entered from below, the time gap is at least `τ_cross`, and the crossing
  distance is at most `d_cross`. Each cross-over costs `C_cross` in the reward. This allows the
  Scarlatti-style left hand over right hand while still forbidding interleaved fingers.
* `free`: no H5. Used only as an ablation, to show what goes wrong without it.

In PIG, about 1.4% of nearby right/left note pairs have the right hand below the left. `strict` is
therefore almost free on real data, and `gated` covers the rest. We will report the exact number.

**No dead ends.** A per-step mask can still trap the agent: an early choice can leave no legal action
a few notes later (for example, the right hand holds a low note while a higher note must go to the
left hand, which H5 forbids). The shield therefore runs a **backward reachability pass** over the
segment. It keeps an action legal only if a fully legal completion of the segment still exists from
the resulting state. This is shielded RL (Alshiekh et al., AAAI 2018) for a deterministic,
finite-horizon problem. It is exact, it costs O(N · |state|) per segment, and it runs once per
episode. Two guarantees follow:

1. **Zero illegal fingerings** on every passage, from every trained or untrained agent.
2. **No stuck episodes:** the mask is never all-false.

Both are unit-tested property tests (§5).

### 2.4 Reward

```
R_t = −α · C_effort(hand, finger)          # stage-1 cost table, per hand (unchanged, mirrored LH)
      −α_h · C_interhand                    # crowding (anchors closer than comfortable), gated cross-over cost
      −β · P_illegal                        # only reachable with the shield off (ablation)
      +λ · B_expert                         # 1 if (hand, finger) matches annotator 1, λ = 0 by default
```

There is a second reward option, `--reward parncutt12`: the paper's 12-rule difficulty *D_t*,
applied per hand. This keeps us directly comparable with Iman & Ahn's *L(s,a)* (their eq. 6).

### 2.5 Oracle

Every cost term depends only on a bounded history: two fingers per hand, plus which note each hand
played last. So the **exact minimum-cost legal fingering** of a segment can be found with dynamic
programming.

* Mode A: the state is (f_L, f_L', f_R, f_R'), at most 625 states per step.
* Mode B: also store the index of the other hand's last note, within a window W = 16. That is an
  approximation, and we will report it as one.

The DP gives the optimal return *J\** for regret (§4). It is also the main non-learned baseline.
We should say plainly why RL is still worth doing:

* The policy generalises without a solver.
* Inference is linear in the number of notes and can run in real time.
* The formulation still works when the cost stops being low-order, for example with fatigue,
  tempo-dependent costs or polyphony.

## 3. Build plan (milestones)

Each milestone is one PR into the stage-2 branch. When the plan is approved, that branch becomes
`main`. `one-hand` stays as it is.

| M | Deliverable | Files | Done when |
| --- | --- | --- | --- |
| **M1** | Two-hand segment extraction, split, stats; fix README test-set count | `fingerly/pig.py` (+`two_hand_segments`), `fingerly/data.py` | coverage table above reproduced by a test; train/val/test by piece, test = 69 multi-annotator pieces |
| **M2** | Legality module: H1–H6, three crossing policies, violation counter | `fingerly/legality.py` | unit tests per rule using hand-written cases |
| **M3** | Shield with backward reachability | `fingerly/shield.py` | property test: 1,000 random segments (real and synthetic) × random-valid policy → 0 violations, mask never empty |
| **M4** | `TwoHandFingeringEnv` (modes A and B), inter-hand cost, `parncutt12` reward | `fingerly/env2.py`, `fingerly/cost.py` | Gymnasium `check_env` passes; sb3-contrib `action_masks()` compatible |
| **M5** | Two-hand DP oracle and baselines (random-valid, greedy, independent-hands stage-1, per-hand HMM) | `fingerly/baselines2.py` | DP ≤ every other method's cost on all test segments |
| **M6** | Agents: masked DQN / DDQN / dueling + n-step (stage 1 code, 10 actions), masked PPO and A2C (sb3-contrib), tabular Q-learning on a coarse state | `fingerly/dqn.py`, `fingerly/agents/` | each learns on a 2-segment toy set to within 5% of DP |
| **M7** | Synthetic two-hand exercises: parallel and contrary-motion scales, Alberti bass + melody, crossing-prone passages, wide leaps | `fingerly/synth.py` | used for training augmentation and the robustness stress sets |
| **M8** | Evaluation harness: every metric in §4, multi-seed runner, rliable-style aggregates, plots | `evaluate2.py`, `fingerly/metrics.py`, `plot.py`, `run_seeds.sh` | one command produces all tables and figures in `results/` |
| **M9** | Experiments: comparison, ablations, robustness | `experiments/*.yaml` | §4 tables filled in |
| **M10** | Visualiser for two hands and the report | `visual/piano/`, `visualize.py`, `docs/report/` | both hands drawn; report follows the course format |

## 4. Performance evaluation plan

**Protocol**

* 5 seeds per method. A seed controls network initialisation, ε-greedy actions, replay sampling,
  the order of training segments and minibatches.
* Evaluate on validation every 10k steps. Keep the best checkpoint by validation return. Run the
  test set **once**.
* Report mean ± std, the interquartile mean and 95% bootstrap confidence intervals (Agarwal et al.,
  NeurIPS 2021). For pairwise claims, use a paired Wilcoxon test across test pieces.
* Hardware, wall-clock time and every hyperparameter go into `results/*.json`.

### 4.1 RL-specific metrics

| Metric | Definition here | Plot / table |
| --- | --- | --- |
| Mean episodic return | `G = Σ γ^t r_{t+1}` on validation segments, normalised per note | learning curve vs environment steps, mean with a 95% CI band, all algorithms |
| Success rate | an episode **succeeds** if (a) it has 0 illegal moves and (b) its effort per note is ≤ 1.10 × the DP optimum for that segment. This is fixed before any run. | success vs steps; final table with CI |
| Cumulative regret | `r_i = C_i^π − C_i^*` against the DP oracle on test segment *i*; `R_N = Σ r_i` | cumulative regret vs test segments, per method (oracle = 0 line) |
| Convergence speed | `N* = min N : Ḡ_N ≥ G_th for K = 3 consecutive evaluations`, with G_th = 95% of the DP-optimal normalised return | annotated threshold plot; "steps to threshold" column |
| Sample efficiency | area under the learning curve, and N* | table |
| Generalization gap | `Δ = J_seen − J_unseen` for (i) train vs test pieces, (ii) leave-one-composer-out, (iii) real PIG vs synthetic stress sets | grouped bar chart per method |
| Stability across seeds | std and interquartile range of final return, plus the width of the CI band | table and CI bands |
| Computational efficiency | training time, inference ms per note, parameter count, peak memory | table |

### 4.2 Problem-specific metrics

| Metric | Goal |
| --- | --- |
| **Illegal-move rate**, split by rule H1–H6 (span, per-hand polyphony, collision, illegal crossing, shift time) | **0** for every shielded method; reported for unshielded baselines |
| Mgen / Mhigh / Msoft (Nakamura et al.), with a note matching only if *hand and finger* both agree | ↑, compared with human leave-one-out agreement (the ceiling) |
| **Hand-assignment accuracy** against the annotators (mode B) | ↑ |
| Effort per note (our cost) and the paper's difficulty L(s,a) (12 Parncutt rules) | ↓; L(s,a) lets us compare with the paper's Table II |
| Hand-crossing rate, inter-hand crowding rate, distribution of anchor distance | crossings close to the human rate, no crowding |
| Thumb-under / finger-over rate, hand-shift rate, weak-finger rate, stretch per note | close to the human rates |

### 4.3 Comparison with benchmark methods

The action space is discrete, so following the course guidance we compare:

* **RL:**
  * tabular Q-learning
  * DQN (the paper's best)
  * Double DQN
  * Dueling Double DQN with n-step returns (ours)
  * masked PPO
  * masked A2C (the paper's on-policy arm)
* **Non-RL:**
  * random-valid (floor)
  * greedy cost
  * DP oracle (cost ceiling)
  * per-hand first-order HMM, trained on labels and given the score's hand split
  * independent hands, which runs the stage-1 one-hand agent on each hand separately. This tests
    whether modelling the hands jointly helps.
* **Paper reproduction:** a plain DQN with the paper's 273-dimensional one-hot state and the
  `parncutt12` reward, on the right hand of the same segments.

For each comparison we explain *why* a method wins or loses. Examples: on-policy methods were weak
in the paper, and DDQN gave no gain there.

### 4.4 Ablation study (one component removed at a time, starting from the full method)

| Variant | Question |
| --- | --- |
| no shield (mask off, penalty β only) | does masking matter, or does the penalty alone teach legality? Expect many crossing and span violations |
| mask without dead-end reachability | how often do agents get stuck, and what does that cost? |
| `--crossing free` / `gated` / `strict` | how H5 changes legality, human match and effort |
| no inter-hand state features | is a joint state needed, or are two independent hands enough? |
| mode A (given split) vs mode B (free split) | value and difficulty of learning hand choice |
| no look-ahead (k = 0) | the benefit of seeing the next notes |
| no n-step / no dueling / no double | contribution of each DQN extension |
| λ = 0 vs λ = 0.5 | the trade-off between imitation and effort |
| stage-1 cost vs `parncutt12` reward | sensitivity to the choice of reward |
| no synthetic augmentation | value of the extra exercises |

### 4.5 Robustness and sensitivity

* **Hand size:** `span_scale` from 0.8 to 1.2 at test time (policy trained at 1.0) and retrained.
* **Tempo:** onsets scaled ×0.5 to ×2 with H6 on. Faster music should make jumps illegal.
* **Transposition:** ±1 to ±6 semitones, which changes the black-key pattern.
* **Timing noise:** jitter on onsets and durations, which affects the H2 and H5 engagement tests.
* **Hyperparameters:** learning rate, γ, n, β, λ, τ_engage and gap_min, one at a time.
* **Stress sets:** crossing-prone passages, hands close together, leaps wider than an octave.

## 5. Testing

* **Unit tests:** H1–H6 on hand-written cases, the mirroring rule, and the parser for two-hand segments.
* **Property tests:**
  * The shield never allows a violation, and the mask is never empty, for random policies on real and synthetic segments.
  * DP cost ≤ greedy cost ≤ random-valid cost on average.
  * Every method's cost ≥ DP cost.
* **Smoke test:** a short DQN run in mode B learns a toy segment to within 5% of DP.

## 6. Risks and how we handle them

| Risk | Plan |
| --- | --- |
| Little two-hand data (~4.4k training and validation notes outside the test pieces) | synthetic exercises (M7); use stage-1 one-hand passages as extra mode-A practice; report the effect in the ablation |
| The reachability pass is slow on long segments | cap the state with window W; cache per segment; measure the cost in the efficiency table |
| Thumb-anchor geometry is a simplification | document it as an assumption; test sensitivity to `off`, `gap_min` and `τ_engage` |
| "Why RL when DP solves it?" | DP is the oracle for regret. The case for RL rests on generalisation, inference speed and higher-order extensions (§2.5); we state this openly in Limitations |
| Parncutt span numbers | check against the 1997 paper before reporting (as noted in `cost.py`) |

## 7. Report structure

Following the course format:

1. Title, abstract, index terms, introduction.
2. Related work: Parncutt 1997, the HMMs of Nakamura et al., DP and VNS/tabu search, Iman & Ahn 2024,
   shielded and masked RL.
3. Problem formulation (§2), proposed method, algorithm box (shielded masked DDQN), system diagram.
4. Experimental setup.
5. Metrics (§4).
6. Results and discussion, comparison, ablation, robustness.
7. Limitations, conclusion, IEEE references.
