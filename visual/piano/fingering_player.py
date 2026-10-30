"""Replay an exported Fingerly fingering on the piano (§12.5).

    python fingering_player.py [PATH.json] [--speed 1.0] [--loop] [--screenshot OUT.png --at SECONDS]

Keys: Space pause/resume, R restart, Right/Left faster/slower (x1.25), Esc quit.
Blue = left hand, orange = right hand.
"""
import argparse
import json

import pygame

import main as piano
from hand import draw_2d_hand

COLORS = {"L": (40, 110, 255), "R": (255, 140, 0)}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def key_of(note):
    """('white' or 'black', key index) of a note; flats become sharps."""
    name = note.get("name") or f"{NAMES[note['pitch'] % 12]}{note['pitch'] // 12 - 1}"
    if name[:2] in FLATS:
        name = FLATS[name[:2]] + name[2:]
    if "#" in name:
        return "black", piano.black_labels.index(name)
    return "white", piano.white_notes.index(name)


def key_box(kind, i, white_rects, black_rects):
    """Outline rectangle of the key and the point where the hand is drawn (as in the user's hand code)."""
    if kind == "white":
        r = white_rects[i]
        return pygame.Rect(r.x, piano.HEIGHT - 100, 35, 100), (r.centerx, r.bottom - 30)
    r = black_rects[i]
    return r, (r.centerx, r.bottom - 20)


def run(data, speed, loop, screenshot=None, at=0.0):
    notes = data["notes"]
    fps = piano.fps
    paused = False
    t, nxt, total, last, current, aw, ab = 0.0, 0, 0.0, {}, None, [], []   # playback state
    while True:
        dt = 1 / fps if screenshot else piano.timer.tick(fps) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    t, nxt, total, last, current, aw, ab = 0.0, 0, 0.0, {}, None, [], []
                elif event.key == pygame.K_RIGHT:
                    speed *= 1.25
                elif event.key == pygame.K_LEFT:
                    speed /= 1.25
        if not paused:
            t += dt * speed

        # start every note whose onset has been reached
        while nxt < len(notes) and notes[nxt]["onset"] - notes[0]["onset"] <= t:
            n = notes[nxt]
            kind, i = key_of(n)
            (piano.white_sounds if kind == "white" else piano.black_sounds)[i].play(0, 1000)
            (aw if kind == "white" else ab).append([i, max(8, int(fps * (n["offset"] - n["onset"]) / speed))])
            prev = last.get(n["hand"])  # notes of one hand starting together form a chord: keep them all
            same = prev and prev[0] == n["onset"]
            last[n["hand"]] = (n["onset"], (prev[1] if same else []) + [(kind, i, n["finger"])])
            total += n["cost"]
            current, nxt = n, nxt + 1
        done = nxt == len(notes)

        piano.screen.fill("gray")
        white_rects, black_rects, aw, ab = piano.draw_piano(aw, ab)
        for hand, (_, keys) in last.items():
            for j, (kind, i, finger) in enumerate(keys):
                rect, target = key_box(kind, i, white_rects, black_rects)
                pygame.draw.rect(piano.screen, COLORS[hand], rect, 4)
                if j == 0:  # one hand drawing per hand, on the chord's first note
                    draw_2d_hand(piano.screen, piano.HEIGHT, [], [], white_rects, black_rects,
                                 active_finger=finger, side=1 if hand == "R" else -1, target=target)
                piano.screen.blit(piano.medium_font.render(str(finger), True, COLORS[hand]), (rect.centerx - 6, 70))
        if current:
            info = (f"{data['policy']} | {current['name']} | hand {current['hand']} | finger {current['finger']}"
                    f" | cost {current['cost']:.1f} | total {total:.1f}")
            labelled = [m for m in notes[:nxt] if m.get("human")]
            if current.get("human"):
                info += f" | human {current['human']}" + ("" if current["human"] == current["finger"] else " (differs)")
            if labelled:
                info += f" | agrees {sum(m['human'] == m['finger'] for m in labelled)}/{len(labelled)}"
            rules = ", ".join(f"{k} {v:g}" for k, v in current["breakdown"].items()) or "no rules fired"
            piano.screen.blit(piano.small_font.render(info, True, "black"), (10, 8))
            piano.screen.blit(piano.small_font.render(rules, True, "black"), (10, 30))
        legend = "Blue = left hand, Orange = right hand" + ("   Done - R to replay, Esc to quit" if done else "")
        piano.screen.blit(piano.small_font.render(legend, True, "black"), (10, 52))
        pygame.display.flip()

        if screenshot and t >= at:
            pygame.image.save(piano.screen, screenshot)
            return
        if done and loop and not any(k[1] > 0 for k in aw + ab):
            t, nxt, total, last, current = 0.0, 0, 0.0, {}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="../../results/fingerings/cross_example__agent.json",
                    help="exported fingering JSON (default: the cross example)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--screenshot")
    ap.add_argument("--at", type=float, default=0.0)
    args = ap.parse_args()
    with open(args.path, encoding="utf-8") as f:
        run(json.load(f), args.speed, args.loop, args.screenshot, args.at)
    pygame.quit()


if __name__ == "__main__":
    main()
