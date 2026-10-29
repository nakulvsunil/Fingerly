# Part A results: Iman & Ahn (2024) reproduction

Rules: `--table1 paper --same-finger zero-span`. Policy scored: **best** (best training episode). Runs per cell: V0/a2c=10, V0/ddqn=10, V0/dqn=10, V0/pg=10, V1/a2c=10, V1/ddqn=10, V1/dqn=10, V1/pg=10, V2/a2c=10, V2/ddqn=10, V2/dqn=10, V2/pg=10, V3/a2c=10, V3/ddqn=10, V3/dqn=10, V3/pg=10.

Each cell is **ours / paper**.

## Table II: best L over runs (lower is better)

| Method | V0 | V1 | V2 | V3 |
| --- | --- | --- | --- | --- |
| DQN | 0.6207 / 0.6207 | 1.3349 / 1.3721 | 3.0588 / 3.1814 | 2.5000 / 2.5441 |
| DDQN | 0.6207 / 0.6207 | 1.3488 / 2.5163 | 3.2206 / 5.2647 | 2.6618 / 3.25 |
| PG | 2.8621 / 8.1034 | 5.1256 / 10.712 | 8.9338 / 14.478 | 6.2206 / 11.294 |
| A2C | 5.5172 / 2.7586 | 6.0140 / 3.893 | 8.1765 / 7.0441 | 10.8824 / 6.6176 |
| FHMM1 | – / 1.0697 | – / 2.3349 | – / 3.6029 | – / 2.832 |
| FHMM2 | – / 0.6207 | – / 1.8884 | – / 3.3235 | – / 2.8028 |
| FHMM3 | – / 0.6207 | – / 2.2512 | – / 3.2941 | – / 2.9859 |
| Human (annotator 1) | 0.6207 (all: 0.6207) / 0.6207 | 1.9581 (all: 1.9581, 2.1395, 1.9349, 1.9070, 1.8279) / 1.8419 | 4.1324 (all: 4.1324, 4.1912) / 4.0441 | 3.1324 (all: 3.1324, 3.0441) / 2.9155 |
| Exact optimum (DP) | 0.6207 / – | 1.2186 / – | 2.6029 / – | 1.8971 / – |

## Tables III–IV: mean ± std of L over runs, and mean run time

| Method | V0 | V1 | V2 | V3 | mean time (s) per run, V0..V3 |
| --- | --- | --- | --- | --- | --- |
| DQN | 0.7759 ± 0.120 / 1.2552 ± 0.905 | 1.3679 ± 0.040 / 1.51 ± 0.17 | 3.3154 ± 0.204 / 3.78 ± 0.30 | 2.8721 ± 0.171 / 3.14 ± 0.53 | 84, 1284, 377, 203 |
| DDQN | 0.9310 ± 0.208 / 0.7000 ± 0.238 | 1.4005 ± 0.051 / 2.92 ± 0.21 | 3.5059 ± 0.174 / 6.05 ± 0.78 | 3.3074 ± 0.462 / 3.76 ± 0.35 | 93, 773, 453, 219 |
| PG | 3.7379 ± 0.656 / 9.7138 ± 1.267 | 6.5874 ± 1.125 / 11.61 ± 0.39 | 10.5007 ± 1.039 / 16.5 ± 0.98 | 7.3647 ± 0.759 / 13.38 ± 1.05 | 15, 97, 41, 22 |
| A2C | 7.5483 ± 1.154 / 3.7448 ± 2.759 | 9.9047 ± 2.088 / 5.08 ± 0.83 | 13.2912 ± 2.652 / 8.94 ± 1.33 | 12.0662 ± 0.925 / 7.75 ± 0.56 | 153, 1319, 841, 366 |

## Table V: match rates vs PIG annotations, averaged over V1–V3 (higher is better)

Each algorithm row is the mean over its runs of the per-piece-averaged rates; `best` is the run with the lowest L on each piece.

| Method | Mgen | Mhigh | Msoft | Mrec |
| --- | --- | --- | --- | --- |
| DQN | 55.86 (best 53.52) / 66.50 | 60.24 (best 58.13) / 68.97 | 69.94 (best 68.36) / 78.52 | 66.86 (best 64.67) / 72.55 |
| DDQN | 55.52 (best 57.09) / 58.23 | 59.17 (best 60.38) / 61.10 | 69.46 (best 71.65) / 70.27 | 66.44 (best 69.34) / 62.29 |
| PG | 29.00 (best 30.43) / 24.20 | 31.05 (best 32.89) / 26.97 | 37.46 (best 38.84) / 34.60 | 34.16 (best 36.13) / 28.16 |
| A2C | 24.82 (best 30.30) / 43.08 | 26.56 (best 33.16) / 46.30 | 32.88 (best 39.56) / 54.18 | 29.39 (best 36.02) / 47.02 |
| Exact optimum (DP) | 62.10 / – | 66.65 / – | 77.15 / – | 73.70 / – |
| FHMM1 | – / 63.50 | – / 66.11 | – / 79.47 | – / 74.22 |
| FHMM2 | – / 61.32 | – / 63.96 | – / 76.61 | – / 70.41 |
| FHMM3 | – / 60.62 | – / 61.34 | – / 75.42 | – / 70.17 |

Per piece (mean over runs | best run):

| Method | Piece | Mgen | Mhigh | Msoft | Mrec | best Mgen | best Mhigh | best Msoft | best Mrec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DQN | V1 | 62.94 | 67.12 | 80.05 | 76.33 | 60.93 | 65.58 | 78.60 | 74.88 |
| DQN | V2 | 49.04 | 52.43 | 64.93 | 61.18 | 48.90 | 54.41 | 64.71 | 60.29 |
| DQN | V3 | 55.59 | 61.18 | 64.85 | 63.09 | 50.74 | 54.41 | 61.76 | 58.82 |
| DDQN | V1 | 62.91 | 66.93 | 80.00 | 76.74 | 64.28 | 67.91 | 81.86 | 78.60 |
| DDQN | V2 | 52.02 | 55.44 | 67.79 | 64.04 | 58.46 | 58.82 | 75.74 | 73.53 |
| DDQN | V3 | 51.62 | 55.15 | 60.59 | 58.53 | 48.53 | 54.41 | 57.35 | 55.88 |
| PG | V1 | 31.07 | 33.21 | 42.37 | 38.00 | 32.84 | 33.95 | 43.72 | 40.00 |
| PG | V2 | 26.43 | 28.46 | 33.53 | 30.81 | 28.31 | 30.88 | 34.56 | 31.62 |
| PG | V3 | 29.49 | 31.47 | 36.47 | 33.68 | 30.15 | 33.82 | 38.24 | 36.76 |
| A2C | V1 | 25.50 | 27.26 | 35.40 | 31.26 | 30.60 | 32.56 | 40.00 | 36.74 |
| A2C | V2 | 25.37 | 26.84 | 33.38 | 30.44 | 37.50 | 41.91 | 49.26 | 46.32 |
| A2C | V3 | 23.60 | 25.59 | 29.85 | 26.47 | 22.79 | 25.00 | 29.41 | 25.00 |

## Claims check (README §A.6)

- **DQN reaches the V0 optimum (−18):** 3/10 runs had a −18 training episode (first at episode 229, median 368; paper: 124); 0/10 final greedy policies score 18.
- **DDQN reaches the V0 optimum (−18):** 1/10 runs had a −18 training episode (first at episode 229, median 229; paper: 202); 0/10 final greedy policies score 18.
- **Ordering DQN ≤ DDQN < A2C < PG on V0** (mean L): DQN 0.776, DDQN 0.931, PG 3.738, A2C 7.548 → does NOT hold
- **Ordering DQN ≤ DDQN < A2C < PG on V1** (mean L): DQN 1.368, DDQN 1.400, PG 6.587, A2C 9.905 → does NOT hold
- **Ordering DQN ≤ DDQN < A2C < PG on V2** (mean L): DQN 3.315, DDQN 3.506, PG 10.501, A2C 13.291 → does NOT hold
- **Ordering DQN ≤ DDQN < A2C < PG on V3** (mean L): DQN 2.872, DDQN 3.307, PG 7.365, A2C 12.066 → does NOT hold
- **DQN Mgen vs FHMM1:** ours 55.86% vs paper's FHMM1 63.5% (paper's DQN 66.5%) → gap -7.64 points.

FHMM1–3 were not run (Nakamura et al.'s programs are not in this repository); their numbers are the paper's.
V0 is a reconstruction from the paper's Fig. 5 (see `data/paper/V0_beyer_op101_no12_fingering.txt`).