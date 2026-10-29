# Part A results: Iman & Ahn (2024) reproduction

Rules: `--table1 paper --same-finger zero-span`. Policy scored: **final** (greedy policy after training). Runs per cell: V0/a2c=10, V0/ddqn=10, V0/dqn=10, V0/pg=10, V1/a2c=10, V1/ddqn=10, V1/dqn=10, V1/pg=10, V2/a2c=10, V2/ddqn=10, V2/dqn=10, V2/pg=10, V3/a2c=10, V3/ddqn=10, V3/dqn=10, V3/pg=10.

Each cell is **ours / paper**.

## Table II: best L over runs (lower is better)

| Method | V0 | V1 | V2 | V3 |
| --- | --- | --- | --- | --- |
| DQN | 0.8966 / 0.6207 | 1.3674 / 1.3721 | 3.3088 / 3.1814 | 2.9853 / 2.5441 |
| DDQN | 0.9310 / 0.6207 | 1.3953 / 2.5163 | 3.4191 / 5.2647 | 3.7353 / 3.25 |
| PG | 3.0000 / 8.1034 | 4.8791 / 10.712 | 9.9191 / 14.478 | 7.0294 / 11.294 |
| A2C | 9.1034 / 2.7586 | 10.8047 / 3.893 | 17.0074 / 7.0441 | 14.8529 / 6.6176 |
| FHMM1 | – / 1.0697 | – / 2.3349 | – / 3.6029 | – / 2.832 |
| FHMM2 | – / 0.6207 | – / 1.8884 | – / 3.3235 | – / 2.8028 |
| FHMM3 | – / 0.6207 | – / 2.2512 | – / 3.2941 | – / 2.9859 |
| Human (annotator 1) | 0.6207 (all: 0.6207) / 0.6207 | 1.9581 (all: 1.9581, 2.1395, 1.9349, 1.9070, 1.8279) / 1.8419 | 4.1324 (all: 4.1324, 4.1912) / 4.0441 | 3.1324 (all: 3.1324, 3.0441) / 2.9155 |
| Exact optimum (DP) | 0.6207 / – | 1.2186 / – | 2.6029 / – | 1.8971 / – |

## Tables III–IV: mean ± std of L over runs, and mean run time

| Method | V0 | V1 | V2 | V3 | mean time (s) per run, V0..V3 |
| --- | --- | --- | --- | --- | --- |
| DQN | 0.9414 ± 0.040 / 1.2552 ± 0.905 | 1.4972 ± 0.161 / 1.51 ± 0.17 | 3.9529 ± 0.697 / 3.78 ± 0.30 | 4.2574 ± 0.832 / 3.14 ± 0.53 | 84, 1284, 377, 203 |
| DDQN | 1.3690 ± 0.744 / 0.7000 ± 0.238 | 1.5758 ± 0.138 / 2.92 ± 0.21 | 4.2618 ± 0.554 / 6.05 ± 0.78 | 4.9882 ± 0.691 / 3.76 ± 0.35 | 93, 773, 453, 219 |
| PG | 5.0655 ± 2.292 / 9.7138 ± 1.267 | 7.9633 ± 2.784 / 11.61 ± 0.39 | 13.3309 ± 2.996 / 16.5 ± 0.98 | 9.2676 ± 2.275 / 13.38 ± 1.05 | 15, 97, 41, 22 |
| A2C | 10.3724 ± 0.978 / 3.7448 ± 2.759 | 12.5135 ± 1.240 / 5.08 ± 0.83 | 18.9037 ± 1.441 / 8.94 ± 1.33 | 15.7147 ± 0.880 / 7.75 ± 0.56 | 153, 1319, 841, 366 |

## Table V: match rates vs PIG annotations, averaged over V1–V3 (higher is better)

Each algorithm row is the mean over its runs of the per-piece-averaged rates; `best` is the run with the lowest L on each piece.

| Method | Mgen | Mhigh | Msoft | Mrec |
| --- | --- | --- | --- | --- |
| DQN | 52.53 (best 55.03) / 66.50 | 57.47 (best 58.78) / 68.97 | 66.40 (best 69.22) / 78.52 | 63.09 (best 66.60) / 72.55 |
| DDQN | 51.24 (best 49.41) / 58.23 | 55.72 (best 54.30) / 61.10 | 64.92 (best 63.15) / 70.27 | 61.50 (best 60.10) / 62.29 |
| PG | 28.56 (best 29.53) / 24.20 | 30.78 (best 30.64) / 26.97 | 36.72 (best 36.79) / 34.60 | 32.97 (best 33.35) / 28.16 |
| A2C | 21.08 (best 22.20) / 43.08 | 23.27 (best 23.99) / 46.30 | 28.85 (best 29.75) / 54.18 | 24.58 (best 26.52) / 47.02 |
| Exact optimum (DP) | 62.10 / – | 66.65 / – | 77.15 / – | 73.70 / – |
| FHMM1 | – / 63.50 | – / 66.11 | – / 79.47 | – / 74.22 |
| FHMM2 | – / 61.32 | – / 63.96 | – / 76.61 | – / 70.41 |
| FHMM3 | – / 60.62 | – / 61.34 | – / 75.42 | – / 70.17 |

Per piece (mean over runs | best run):

| Method | Piece | Mgen | Mhigh | Msoft | Mrec | best Mgen | best Mhigh | best Msoft | best Mrec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DQN | V1 | 63.72 | 68.37 | 80.98 | 77.07 | 62.51 | 66.05 | 80.47 | 76.28 |
| DQN | V2 | 45.70 | 50.22 | 61.62 | 57.35 | 48.90 | 51.47 | 65.44 | 61.76 |
| DQN | V3 | 48.16 | 53.82 | 56.62 | 54.85 | 53.68 | 58.82 | 61.76 | 61.76 |
| DDQN | V1 | 61.73 | 66.28 | 79.02 | 75.40 | 59.63 | 65.12 | 77.67 | 74.42 |
| DDQN | V2 | 46.76 | 50.59 | 61.47 | 57.21 | 41.54 | 44.85 | 57.35 | 52.94 |
| DDQN | V3 | 45.22 | 50.29 | 54.26 | 51.91 | 47.06 | 52.94 | 54.41 | 52.94 |
| PG | V1 | 28.71 | 31.16 | 40.60 | 34.51 | 22.05 | 22.79 | 30.23 | 26.51 |
| PG | V2 | 29.26 | 31.32 | 35.00 | 33.38 | 34.93 | 35.29 | 41.91 | 39.71 |
| PG | V3 | 27.72 | 29.85 | 34.56 | 31.03 | 31.62 | 33.82 | 38.24 | 33.82 |
| A2C | V1 | 20.55 | 22.98 | 29.63 | 24.84 | 12.56 | 15.35 | 22.33 | 20.00 |
| A2C | V2 | 22.24 | 25.37 | 30.88 | 26.10 | 28.31 | 30.15 | 33.09 | 31.62 |
| A2C | V3 | 20.44 | 21.47 | 26.03 | 22.79 | 25.74 | 26.47 | 33.82 | 27.94 |

## Claims check (README §A.6)

- **DQN reaches the V0 optimum (−18):** 3/10 runs had a −18 training episode (first at episode 229, median 368; paper: 124); 0/10 final greedy policies score 18.
- **DDQN reaches the V0 optimum (−18):** 1/10 runs had a −18 training episode (first at episode 229, median 229; paper: 202); 0/10 final greedy policies score 18.
- **Ordering DQN ≤ DDQN < A2C < PG on V0** (mean L): DQN 0.941, DDQN 1.369, PG 5.066, A2C 10.372 → does NOT hold
- **Ordering DQN ≤ DDQN < A2C < PG on V1** (mean L): DQN 1.497, DDQN 1.576, PG 7.963, A2C 12.513 → does NOT hold
- **Ordering DQN ≤ DDQN < A2C < PG on V2** (mean L): DQN 3.953, DDQN 4.262, PG 13.331, A2C 18.904 → does NOT hold
- **Ordering DQN ≤ DDQN < A2C < PG on V3** (mean L): DQN 4.257, DDQN 4.988, PG 9.268, A2C 15.715 → does NOT hold
- **DQN Mgen vs FHMM1:** ours 52.53% vs paper's FHMM1 63.5% (paper's DQN 66.5%) → gap -10.97 points.

FHMM1–3 were not run (Nakamura et al.'s programs are not in this repository); their numbers are the paper's.
V0 is a reconstruction from the paper's Fig. 5 (see `data/paper/V0_beyer_op101_no12_fingering.txt`).