# Two-hand data coverage (README §C.4.8)

`chord_tol` 0.03 s, `overlap_tol` 0.05 s, `rest_gap` 1.5 s, segments 16–256 events. Notes, hands and onsets from annotator 1; all 150 PIG pieces.

| Hand | Setting | Notes fingered | Segments | Mean segment length | Chord events | Held notes |
| --- | --- | --- | --- | --- | --- | --- |
| R | 1: Part B monophonic filter | 9739/26446 (36.8%) | 204 | 47.7 | 0.0% | 0.0% |
| R | 2: overlap tolerance only | 12610/26446 (47.7%) | 186 | 67.8 | 0.0% | 3.6% |
| R | 3: overlap tolerance + chords | 26324/26446 (99.5%) | 173 | 117.4 | 21.2% | 6.4% |
| L | 1: Part B monophonic filter | 6865/22278 (30.8%) | 133 | 51.6 | 0.0% | 0.0% |
| L | 2: overlap tolerance only | 9903/22278 (44.5%) | 145 | 68.3 | 0.0% | 2.3% |
| L | 3: overlap tolerance + chords | 21914/22278 (98.4%) | 175 | 93.6 | 25.1% | 4.2% |

## Training events by chord size

| Size | Events |
| --- | --- |
| 1 | 18890 |
| 2 | 3876 |
| 3 | 1544 |
| 4 | 270 |
| 5 | 4 |

Pipeline log over all pieces: 4430 legato overlaps made sequential, 2627 held notes, 0 chords cut to 5 notes.

## Split

```
train  108 pieces   139 segments   24584 events   32374 notes
val     12 pieces    14 segments    1972 events    2981 notes
test    30 pieces    57 segments   10299 events   13212 notes
```

## Within-hand histograms (training pieces)

| Onset difference (s) | Count |
| --- | --- |
| 0–0.01 | 0 |
| 0.01–0.02 | 1 |
| 0.02–0.03 | 0 |
| 0.03–0.05 | 67 |
| 0.05–0.08 | 287 |
| 0.08–0.12 | 1552 |
| 0.12–inf | 30275 |

| Overlap length (s) | Count |
| --- | --- |
| 0–0.01 | 298 |
| 0.01–0.02 | 985 |
| 0.02–0.03 | 274 |
| 0.03–0.05 | 298 |
| 0.05–0.08 | 55 |
| 0.08–0.12 | 43 |
| 0.12–0.2 | 125 |
| 0.2–0.5 | 820 |
| 0.5–inf | 906 |
