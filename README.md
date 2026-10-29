# Fingerly — stage 1: one hand, monophonic

Model-free deep RL for piano fingering on the PIG dataset. This stage fingers **one hand at a
time** (right hand by default, `--hand L` for the left) on **monophonic passages** only: every
note gets a finger 1–5 from a masked DQN / Double DQN agent trained on an ergonomic reward.
It is stage 1 of the plan in the proposal ("rebuild the right-hand DQN of Iman & Ahn"); the
hand choice and the 10-action space come in stage 2.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # a few minutes; includes a short DQN training run
```

Get PIG from <http://beam.kisarazu.ac.jp/research/PianoFingeringDataset/> (it needs the
usage agreement), unzip it, e.g. to `data/PianoFingeringDataset_v1.2`. `--pig-dir` can point
at the root or straight at `FingeringFiles/`.

No dataset yet? `python -m fingerly.fake_pig --out data/fake_pig` writes a fake dataset in
PIG format so you can test the whole pipeline; the numbers it gives are meaningless.

## Run

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

Useful switches: `--no-double --no-dueling` (plain DQN as in Iman & Ahn), `--lam 0.5`
(expert-match bonus), `--synthetic 500` (add scales/arpeggios/melodies), `--no-mask`
(ablation: the agent can make impossible moves and pays the penalty), `--span-scale 0.85`
(smaller hand), `--select-by match` (keep the checkpoint with the best validation Mgen
instead of the best validation reward), `--steps`, `--lr`, `--n-step`.

## Layout

| File | What it does |
| --- | --- |
| `fingerly/pig.py` | PIG reader (pitch → MIDI, finger sign → hand, `3_1` → 3), monophonic filter, train/val/test split |
| `fingerly/cost.py` | Cost table from the proposal + Parncutt et al. (1997) span limits, action mask |
| `fingerly/env.py` | Gymnasium environment (`action_masks()` is sb3-contrib compatible) |
| `fingerly/dqn.py` | Masked DQN / Double DQN, dueling head, n-step returns, replay buffer, training loop |
| `fingerly/baselines.py` | Random-valid, greedy, optimal-cost DP, first-order HMM |
| `fingerly/metrics.py` | Mgen / Mhigh / Msoft, human agreement, playability metrics |
| `fingerly/synth.py` | Label-free synthetic exercises |
| `fingerly/fake_pig.py` | Fake PIG-format data for testing |
| `fingerly/export.py` | Fingers a PIG file and exports it (per-note cost and rules) as JSON for the player |
| `visual/piano/` | pygame piano, hand drawing and `fingering_player.py` (plays an exported JSON) |
| `train.py`, `evaluate.py`, `predict.py`, `plot.py`, `visualize.py` | Command-line tools |

## Design

**Data.** For each piece the note list and the hand of each note come from annotator 1.
A hand's notes form a passage while each note starts after every earlier note of that hand
has ended; chords and notes played under a held note end the passage and are dropped.
Passages need ≥ 8 notes (`--min-len`). Test = the pieces fingered by several pianists
(30 in PIG); of the rest, 12 pieces (seeded) are validation and the others training.
Pieces are never split. `train.py` prints how many notes the filter keeps.

**State** (24 numbers with look-ahead 3): current note (distance from middle C / 24, black?);
next 3 notes (interval / 12, black?, exists?); the hand (played yet?, interval from its last
key, last finger one-hot, interval from its second-to-last key, second-to-last finger
one-hot). The second-to-last finger follows Iman & Ahn's "last two notes and fingers".
Left-hand intervals are mirrored so both hands share one cost model.

**Action** finger 1–5. **Reward** `−α·C_effort − β·P_impossible + λ·B_expert` with the
proposal's table: 1 per semitone wider or narrower than comfortable, 1 for finger 4/5,
1 for thumb on black, 1 for a crossing, 2 for the same finger on a different key, 10 for
an impossible move. Comfortable and practical spans per finger pair are Parncutt's Table 1
(check the numbers against the paper before you cite them).

Three rules were needed that the proposal does not spell out — worth stating in the report:

1. **Out of reach → hand shift.** If no other finger can reach the next key from the
   current one, the hand must jump; then every finger is allowed and the move costs 2
   (same as "same finger, different key"). Without this, large leaps would leave only one
   legal finger.
2. **Repeated note with a new finger** is always allowed and scored by the comfort rule
   (e.g. 2→3 on the same key costs 1).
3. **First note** pays only the per-note costs (weak finger, thumb on black).

With the mask on, the impossible penalty is never charged; it matters only for `--no-mask`.

**Optimal-cost DP.** The reward depends only on the previous finger, the current finger and
the interval, so the minimum-cost fingering of a passage can be computed exactly by dynamic
programming. That is the `optimal-cost (DP)` baseline. The agent optimises the same thing
(γ = 0.99), so DP is the ceiling for reward, and the gap between agent and DP shows how
well it learned. Expect a reviewer to ask "why RL if DP solves it?": the answer is that this
reward is the stage-1 sanity check; RL earns its place once the reward stops being
first-order (imitation term with λ, two hands, look-ahead-dependent costs, polyphony).

**Baselines.** `hmm-1st-order` is a simplified, first-order, semitone version of Nakamura et
al.'s HMM trained on the training labels — a stand-in for the 3rd-order HMM, not a
reproduction. PianoPlayer and the published HMM are not included; run them separately on
the same passages, because published PIG numbers include chords and both hands.

**Metrics.** A note matches an annotator only if the annotator gave it to the same hand and
the same finger. Mgen / Mhigh / Msoft are computed per piece, then averaged over pieces.
"human agreement (LOO)" scores each annotator against the others on the same passages — the
ceiling to compare with. Playability: effort per note, impossible-move rate, crossing rate,
shift rate, weak-finger rate.
