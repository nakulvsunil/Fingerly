# Reproducing Iman & Ahn (2024)

*A Model-Free Deep Reinforcement Learning Approach to Piano Fingering Generation*, IEEE CAI 2024.

```bash
python reproduce_paper.py --pig-dir data/PianoFingeringDataset_v1.2 --workers 4   # ~7 h on 4 CPU cores
python -m pytest tests/test_paper.py
```

All the numbers are in [`results/paper/report.md`](../results/paper/report.md). The raw runs are in
`results/paper/raw/`, one JSON file per run with its episode rewards, final fingering and run time.

## Setup

**Same as the paper**

* **Algorithms:** DQN, DDQN, REINFORCE with baseline (PG), A2C.
* **Pieces:**
  * V0: Beyer Op. 101 No. 12, 29 notes, transcribed from the paper's Fig. 5.
  * V1: Chopin Waltz Op. 64 No. 1, 215 notes, PIG piece 025.
  * V2: Für Elise, 136 notes, PIG piece 061.
  * V3: Scarlatti K. 208, 68 notes, PIG piece 140.
* **Runs:** 10 per algorithm and piece, 500 episodes each.
* **Networks and training:** 512-256-64 ReLU, Adam with learning rate 1e-3, γ = 0.99, MSE loss.
* **DQN and DDQN:** replay memory of 10,000 steps, batch 32, target network synced every 100 steps.
* **State:** the one-hot (n_{t-2}, a_{t-2}, n_{t-1}, a_{t-1}, n_t).
* **Reward:** minus the 12-rule difficulty.

**Checks that the setup matches**

* **The 12 rules.** They reproduce the paper's three scores on V0: 18 and 80 points exactly, and
  234 against 235 for PG.
* **Note counts.** The paper's L values times 215, 136 and 68 give whole numbers, so our V1–V3
  have the same number of notes as the paper's.

**Assumptions where the paper says nothing**

* ε decays linearly from 1 to 0.01 over the first 100 episodes.
* The fingering we score is the greedy one after training.
* PG updates once per episode, A2C once per note.
* The state has 274 inputs; the paper says 273, but its own encoding adds up to 274.
* Mrec is approximate (switching annotator costs one match).
* Hardware is a CPU, not the paper's RTX 2080, so run times are not comparable.

## Results at a glance (difficulty L, lower is better)

| | V0 ours / paper | V1 ours / paper | V2 ours / paper | V3 ours / paper |
|---|---|---|---|---|
| DQN best | **0.62 / 0.62** | **1.00** / 1.37 | **1.88** / 3.18 | 2.65 / **2.54** |
| DQN mean ± std | 1.27±0.43 / 1.26±0.91 | **1.15±0.11** / 1.51±0.17 | **2.32±0.24** / 3.78±0.30 | 3.21±0.25 / 3.14±0.53 |
| DDQN best | **0.62 / 0.62** | **1.02** / 2.52 | **1.90** / 5.26 | **2.56** / 3.25 |
| PG best | **1.45** / 8.10 | **4.27** / 10.71 | **4.03** / 14.48 | **4.94** / 11.29 |
| A2C best | 5.59 / **2.76** | 8.10 / **3.89** | 11.20 / **7.04** | 9.88 / **6.62** |
| Exact optimum (ours) | 0.62 | 0.89 | 1.49 | 1.66 |

Match with the human fingerings (Mgen, best run, mean over V1–V3):

| Method | Ours | Paper |
|---|---|---|
| DQN | 58.4% | 66.5% |
| DDQN | 57.4% | 58.2% |
| PG | 31.2% | 24.2% |
| A2C | 24.3% | 43.1% |

## What reproduces

1. **Off-policy beats on-policy.** This is the paper's main claim. DQN and DDQN are far better than PG
   and A2C on every piece, in both the best and the mean result, and in human match.
2. **DDQN does not improve on DQN.** The two are within noise of each other everywhere: V1
   1.15 vs 1.17, V2 2.32 vs 2.39, V3 3.21 vs 3.36. On V0, DDQN is slightly worse on average.
3. **Both DQN variants find the composer's fingering on Beyer** (L = 0.6207), as in Fig. 5.
4. **The qualitative failure modes are the same.** A2C locks into a local optimum within the first
   few episodes and then stays flat (Fig. 4: A2C "tends to achieve the local maximum at every run").
   PG learns slowly and is still improving at episode 500.
5. **Human match ranks the same:** DQN ≥ DDQN ≫ PG, A2C. DQN's Mgen is within about 1 point of
   the exact optimum's 59.2%.

## What does not reproduce

1. **Our DQN/DDQN results are better than the paper's on V1 and V2.** The gaps are large: V2 DDQN
   is 1.90 against 5.26. On V3 we match the paper. Two causes are likely:
   * our ε schedule, which the paper doesn't give;
   * the paper's DDQN numbers may come from undertrained runs. Its own Table IV shows a much higher
     DDQN mean.
2. **PG is much better here** (1.45 vs 8.10 on V0), even though PG ranks low in both. The paper's PG
   curve *falls* over training (Fig. 4), which suggests something specific to their implementation,
   such as an unnormalised advantage. Ours improves slowly but steadily.
3. **A2C is worse here** (8.10 vs 3.89 on V1). Our A2C collapses almost immediately. The per-note
   rewards are large (−D can reach −20 or worse), so the first updates saturate the softmax. The
   paper's A2C collapses too, just less badly. Reward scaling or an entropy bonus would fix this,
   but the paper uses neither, so we didn't add them.
4. **Our human match is lower** (DQN Mgen 58% vs 66.5%), even though our fingerings are *easier*.
   Under these 12 rules, even the exact minimum-difficulty fingering matches the pianists only 59%
   of the time. The paper's 66.5% is therefore hard to explain by difficulty alone. They may have
   scored a different fingering, for example the best episode seen during training, or computed
   the match differently.
5. **The human scores differ on pieces with black keys.** The annotators' fingerings score
   V2 3.51–3.55 vs 4.04 in the paper and V1 1.65–1.87 vs 1.84. Beyer has no black keys and matches
   exactly. So the black-key rules (thumb, 4 and 5 on black) are probably applied a little
   differently, and absolute numbers on V1–V3 carry that uncertainty.
6. **Run times are not comparable** (our CPU vs their GPU). The ratios between algorithms agree:
   PG is cheapest, and DQN and DDQN cost about the same.

## New finding: how far from optimal?

Every rule looks back at most two notes, so dynamic programming finds the **exact** minimum
difficulty in well under a second. The paper never reports this optimum.

| Piece | Paper's best agent (DQN) | Exact optimum |
|---|---|---|
| V1 | 1.37 | 0.89 |
| V2 | 3.18 | 1.49 |
| V3 | 2.54 | 1.66 |

The paper's best agent is therefore 54–113% above the optimum. Ours is 12% above on V1, 27% on V2
and 59% on V3. The paper also argues that "the agent does not need ground truth" and "beats the
HMMs". Both claims hold, but a 0.1-second exact solver beats every agent. This is the "why RL?"
question the stage-2 plan addresses (docs/PLAN.md §2.5).
