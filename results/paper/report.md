# Reproduction of Iman & Ahn (2024): results

Runs found per (environment, algorithm): [10]. Ours = this repo, CPU (Intel Xeon 2.8 GHz, 1 thread per run). Paper = published numbers.

## Table II: difficulty level L(s,a) of the best run (lower is better)

| Method | V0 ours | V0 paper | V1 ours | V1 paper | V2 ours | V2 paper | V3 ours | V3 paper |
|---|---|---|---|---|---|---|---|---|
| DQN | 0.6207 | 0.6207 | 0.9953 | 1.3721 | 1.8824 | 3.1814 | 2.6471 | 2.5441 |
| DDQN | 0.6207 | 0.6207 | 1.0186 | 2.5163 | 1.8971 | 5.2647 | 2.5588 | 3.2500 |
| PG | 1.4483 | 8.1034 | 4.2651 | 10.7120 | 4.0294 | 14.4780 | 4.9412 | 11.2940 |
| A2C | 5.5862 | 2.7586 | 8.0977 | 3.8930 | 11.1985 | 7.0441 | 9.8824 | 6.6176 |
| FHMM1 [9] | – | 1.0697 | – | 2.3349 | – | 3.6029 | – | 2.8320 |
| FHMM2 [9] | – | 0.6207 | – | 1.8884 | – | 3.3235 | – | 2.8028 |
| FHMM3 [9] | – | 0.6207 | – | 2.2512 | – | 3.2941 | – | 2.9859 |
| Human (annotators, min–max) | 0.6207–0.6207 | 0.6207 | 1.6465–1.8698 | 1.8419 | 3.5074–3.5515 | 4.0441 | 2.4853–2.8235 | 2.9155 |
| **Optimal (exact DP, ours)** | **0.6207** | – | **0.8884** | – | **1.4853** | – | **1.6618** | – |

## Tables III and IV: mean ± std of L over runs, and run time per run

| Method | Env | L ours | L paper | time ours (s, CPU) | time paper (s, GPU) |
|---|---|---|---|---|---|
| DQN | V0 | 1.2724 ± 0.4300 | 1.2552 ± 0.905 | 124 ± 3 | 51.6 |
| DQN | V1 | 1.1521 ± 0.1143 | 1.51 ± 0.17 | 1072 ± 32 | 424 |
| DQN | V2 | 2.3206 ± 0.2431 | 3.78 ± 0.3 | 629 ± 19 | 294 |
| DQN | V3 | 3.2059 ± 0.2462 | 3.14 ± 0.53 | 293 ± 7 | 136 |
| DDQN | V0 | 1.4103 ± 0.6714 | 0.7 ± 0.238 | 130 ± 3 | 55.7 |
| DDQN | V1 | 1.1660 ± 0.1170 | 2.92 ± 0.21 | 1171 ± 15 | 458 |
| DDQN | V2 | 2.3875 ± 0.2966 | 6.05 ± 0.78 | 655 ± 7 | 288 |
| DDQN | V3 | 3.3603 ± 0.5260 | 3.76 ± 0.35 | 310 ± 8 | 146 |
| PG | V0 | 2.9966 ± 1.7341 | 9.7138 ± 1.267 | 12 ± 0 | 9.8 |
| PG | V1 | 6.5614 ± 1.3717 | 11.61 ± 0.39 | 52 ± 2 | 60 |
| PG | V2 | 9.6750 ± 2.9053 | 16.5 ± 0.98 | 34 ± 1 | 43 |
| PG | V3 | 8.4456 ± 2.1684 | 13.38 ± 1.05 | 22 ± 1 | 21 |
| A2C | V0 | 7.6552 ± 1.1016 | 3.7448 ± 2.759 | 232 ± 9 | 45.1 |
| A2C | V1 | 8.2326 ± 0.3134 | 5.08 ± 0.83 | 2003 ± 50 | 345 |
| A2C | V2 | 12.9566 ± 1.8084 | 8.94 ± 1.33 | 1141 ± 25 | 237 |
| A2C | V3 | 10.4706 ± 0.6033 | 7.75 ± 0.56 | 540 ± 14 | 115 |

## Table V: match rate with the human fingerings (%), best run per piece, mean over V1–V3

| Method | Mgen ours | Mgen paper | Mhigh ours | Mhigh paper | Msoft ours | Msoft paper | Mrec* ours | Mrec paper |
|---|---|---|---|---|---|---|---|---|
| DQN | 58.39 | 66.50 | 63.47 | 68.97 | 70.51 | 78.52 | 66.54 | 72.55 |
| DDQN | 57.37 | 58.23 | 60.86 | 61.10 | 69.89 | 70.27 | 65.74 | 62.29 |
| PG | 31.21 | 24.20 | 32.20 | 26.97 | 39.61 | 34.60 | 34.20 | 28.16 |
| A2C | 24.32 | 43.08 | 25.64 | 46.30 | 30.73 | 54.18 | 26.20 | 47.02 |
| FHMM1 [9] | – | 63.50 | – | 66.11 | – | 79.47 | – | 74.22 |
| FHMM2 [9] | – | 61.32 | – | 63.96 | – | 76.61 | – | 70.41 |
| FHMM3 [9] | – | 60.62 | – | 61.34 | – | 75.42 | – | 70.17 |
| **Optimal (exact DP, ours)** | 59.16 | – | 63.69 | – | 72.27 | – | 68.58 | – |

*Mrec: switching annotator costs one match; the exact definition could not be checked.

## Where the difficulty comes from (best DQN run vs exact optimum, points per rule)

| Env | Fingering | stretch | small span | large span | position change count | position change size | weak finger | three four five | three to four | four on black | thumb on black | five on black | thumb passing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | best DQN | 0 | 0 | 0 | 0 | 0 | 9 | 6 | 3 | 0 | 0 | 0 | 0 |
| V0 | optimal | 0 | 0 | 0 | 0 | 0 | 9 | 6 | 3 | 0 | 0 | 0 | 0 |
| V1 | best DQN | 18 | 89 | 18 | 7 | 7 | 31 | 2 | 2 | 0 | 37 | 0 | 3 |
| V1 | optimal | 24 | 75 | 19 | 0 | 0 | 30 | 1 | 0 | 0 | 37 | 2 | 3 |
| V2 | best DQN | 32 | 96 | 57 | 5 | 13 | 36 | 1 | 0 | 0 | 5 | 0 | 11 |
| V2 | optimal | 24 | 60 | 50 | 2 | 4 | 42 | 8 | 1 | 0 | 5 | 0 | 6 |
| V3 | best DQN | 20 | 53 | 13 | 9 | 13 | 14 | 2 | 0 | 1 | 40 | 8 | 7 |
| V3 | optimal | 8 | 33 | 9 | 0 | 0 | 10 | 2 | 2 | 1 | 38 | 2 | 8 |
