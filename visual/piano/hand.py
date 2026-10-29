# import pygame

# # Unique colours for each finger 
# # Thumb: Red, Index: Blue, Middle: Green, Ring: Orange, Pinky: Yellow
# FINGER_COLORS = [
#     (255, 50, 50),   # 0: Thumb 
#     (50, 50, 255),   # 1: Index 
#     (50, 255, 50),   # 2: Middle 
#     (255, 165, 0),   # 3: Ring 
#     (255, 255, 50)   # 4: Pinky 
# ]

# def draw_2d_hand(screen, screen_height, active_whites, active_blacks, white_keys, black_keys, active_finger=0):
#     # Safely convert PyTorch tensors to raw Python integers
#     try:
#         active_finger = int(active_finger)
#     except:
#         active_finger = 0

#     # Auto-normalize the AI's action to index 0-4
#     finger_idx = active_finger - 1 if active_finger > 0 else active_finger
#     finger_idx = max(0, min(4, finger_idx)) 

#     target_x = None
#     target_y = None
    
#     # 1. Get exact position of pressed black key
#     if len(active_blacks) > 0:
#         index = active_blacks[-1][0] 
#         if index < len(black_keys):
#             target_x = black_keys[index].centerx
#             target_y = black_keys[index].bottom - 20
            
#     # 2. Get exact position of pressed white key
#     elif len(active_whites) > 0:
#         index = active_whites[-1][0] 
#         if index < len(white_keys):
#             target_x = white_keys[index].centerx
#             target_y = white_keys[index].bottom - 30

#     # 3. Draw the hand dynamically
#     if target_x is not None and target_y is not None:
#         finger_offsets = [-70, -35, 0, 35, 70]
        
#         palm_x = target_x - finger_offsets[finger_idx]
#         palm_y = screen_height - 330 
        
#         # Draw Palm
#         pygame.draw.circle(screen, (220, 180, 160), (palm_x, palm_y), 25)
        
#         # Draw Fingers
#         for i, offset in enumerate(finger_offsets):
#             finger_x = palm_x + offset
#             finger_y = target_y if i == finger_idx else target_y - 35 
            
#             # Draw bone line
#             pygame.draw.line(screen, (220, 180, 160), (palm_x, palm_y), (finger_x, finger_y), 8)
            
#             # Apply specific colour to the active finger, and skin colour to resting fingers
#             tip_color = FINGER_COLORS[i] if i == finger_idx else (220, 180, 160)
#             pygame.draw.circle(screen, tip_color, (finger_x, finger_y), 10)
            
#             # Draw a white outline on the active finger so it pops out visually
#             if i == finger_idx:
#                 pygame.draw.circle(screen, (255, 255, 255), (finger_x, finger_y), 12, 2)

import pygame
import math

# Unique colours for each finger
# Thumb: Red, Index: Blue, Middle: Green, Ring: Orange, Pinky: Yellow
FINGER_COLORS = [
    (255, 50, 50),   # 0: Thumb
    (50, 50, 255),   # 1: Index
    (50, 255, 50),   # 2: Middle
    (255, 165, 0),   # 3: Ring
    (255, 255, 50)   # 4: Pinky
]

SKIN = (222, 184, 155)
SKIN_SHADOW = (195, 155, 130)
SKIN_HIGHLIGHT = (238, 205, 180)
OUTLINE = (120, 85, 65)

# Relative finger lengths (thumb, index, middle, ring, pinky) — middle is longest
FINGER_LENGTHS = [0.62, 0.92, 1.0, 0.94, 0.78]
# Relative horizontal spread from palm center at the knuckle line
FINGER_SPREAD = [-95, -48, -16, 16, 48]

# Vertical placement of the hand: distance from the bottom of the screen.
# Lower number = hand sits lower on screen.
HAND_VERTICAL_OFFSET = -30

# Fixed side the thumb sits on relative to the palm (-1 = left, 1 = right).
# This must NOT be derived from finger_idx — doing so causes the hand to
# visually flip/invert depending on which finger is active.
THUMB_SIDE = -1


def _draw_tapered_segment(screen, p1, p2, r1, r2, color, outline=True):
    """Draw a finger bone as a tapered quad (wide at base, narrow at tip) with rounded caps."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    nx, ny = -dy / length, dx / length  # perpendicular unit vector

    quad = [
        (p1[0] + nx * r1, p1[1] + ny * r1),
        (p2[0] + nx * r2, p2[1] + ny * r2),
        (p2[0] - nx * r2, p2[1] - ny * r2),
        (p1[0] - nx * r1, p1[1] - ny * r1),
    ]
    pygame.draw.polygon(screen, color, quad)
    pygame.draw.circle(screen, color, (int(p1[0]), int(p1[1])), int(r1))
    pygame.draw.circle(screen, color, (int(p2[0]), int(p2[1])), int(r2))

    if outline:
        pygame.draw.circle(screen, OUTLINE, (int(p1[0]), int(p1[1])), int(r1), 1)
        pygame.draw.circle(screen, OUTLINE, (int(p2[0]), int(p2[1])), int(r2), 1)
        pygame.draw.line(screen, OUTLINE, quad[0], quad[1], 1)
        pygame.draw.line(screen, OUTLINE, quad[2], quad[3], 1)


def _draw_finger(screen, base, angle, length, base_width, tip_width, color, bend_amount, highlight):
    """
    Draw a 3-segment finger (proximal, middle, distal phalanx) curling slightly
    toward the target, so it reads as a real finger rather than a straight line.
    angle: radians, direction the finger points (from base toward tip), where 0 = straight up.
    bend_amount: 0 = flat/straight, 1 = fully curled at the tip (used for the pressing finger).
    """
    seg_fracs = [0.42, 0.33, 0.25]  # proportion of total length per segment
    widths = [
        (base_width, base_width * 0.82),
        (base_width * 0.82, base_width * 0.62),
        (base_width * 0.62, tip_width),
    ]

    points = [base]
    cur_angle = angle
    x, y = base

    for i, frac in enumerate(seg_fracs):
        seg_len = length * frac
        # Each successive joint bends a bit more if bend_amount > 0 (knuckle curl)
        cur_angle += math.radians(bend_amount * (10 + i * 6))
        x = x + math.sin(cur_angle) * seg_len
        y = y - math.cos(cur_angle) * seg_len
        points.append((x, y))

    for i in range(3):
        r1, r2 = widths[i]
        _draw_tapered_segment(screen, points[i], points[i + 1], r1, r2, color if highlight and i == 2 else SKIN)

    # Draw joint knuckle dots for the two interior joints (subtle crease lines)
    for i in (1, 2):
        px, py = points[i]
        pygame.draw.circle(screen, SKIN_SHADOW, (int(px), int(py)), int(widths[i - 1][1] * 0.9), 1)

    # Fingertip: colored if this is the active finger, skin-toned otherwise
    tip_x, tip_y = points[-1]
    final_color = color if highlight else SKIN
    pygame.draw.circle(screen, final_color, (int(tip_x), int(tip_y)), int(tip_width) + 2)
    if highlight:
        pygame.draw.circle(screen, (255, 255, 255), (int(tip_x), int(tip_y)), int(tip_width) + 4, 2)
    pygame.draw.circle(screen, OUTLINE, (int(tip_x), int(tip_y)), int(tip_width) + 2, 1)

    return points[-1]


def draw_2d_hand(screen, screen_height, active_whites, active_blacks, white_keys, black_keys,
                 active_finger=0, side=1, target=None):
    # side: 1 = right hand (current behaviour), -1 = left hand (mirror image)
    # target: optional (x, y) key position; if given, skip the active_whites/active_blacks lookup
    # Safely convert PyTorch tensors to raw Python integers
    try:
        active_finger = int(active_finger)
    except Exception:
        active_finger = 0

    # Auto-normalize the AI's action to index 0-4
    finger_idx = active_finger - 1 if active_finger > 0 else active_finger
    finger_idx = max(0, min(4, finger_idx))

    target_x = None
    target_y = None

    if target is not None:
        target_x, target_y = target
    # 1. Get exact position of pressed black key
    elif len(active_blacks) > 0:
        index = active_blacks[-1][0]
        if index < len(black_keys):
            target_x = black_keys[index].centerx
            target_y = black_keys[index].bottom - 20

    # 2. Get exact position of pressed white key
    elif len(active_whites) > 0:
        index = active_whites[-1][0]
        if index < len(white_keys):
            target_x = white_keys[index].centerx
            target_y = white_keys[index].bottom - 30

    # 3. Draw the hand dynamically
    if target_x is None or target_y is None:
        return

    finger_offsets = [o * side for o in FINGER_SPREAD]
    thumb_side = THUMB_SIDE * side
    palm_x = target_x - finger_offsets[finger_idx]
    palm_y = screen_height - HAND_VERTICAL_OFFSET

    palm_w, palm_h = 100, 90

    # --- Palm: rounded trapezoid (narrower at wrist, wider at knuckles) ---
    knuckle_y = palm_y - palm_h * 0.5
    wrist_y = palm_y + palm_h * 0.5
    palm_poly = [
        (palm_x - palm_w * 0.32, wrist_y),          # wrist left
        (palm_x - palm_w * 0.5, knuckle_y + 10),     # knuckle left
        (palm_x - palm_w * 0.5, knuckle_y),
        (palm_x + palm_w * 0.5, knuckle_y),
        (palm_x + palm_w * 0.5, knuckle_y + 10),     # knuckle right
        (palm_x + palm_w * 0.32, wrist_y),          # wrist right
    ]
    pygame.draw.polygon(screen, SKIN, palm_poly)
    pygame.draw.polygon(screen, OUTLINE, palm_poly, 2)

    # Subtle palm shading (heel of palm)
    pygame.draw.ellipse(
        screen, SKIN_SHADOW,
        (palm_x - palm_w * 0.28, wrist_y - 18, palm_w * 0.56, 26),
        0,
    )
    # Knuckle highlight line
    pygame.draw.line(screen, SKIN_HIGHLIGHT, (palm_x - palm_w * 0.45, knuckle_y + 4),
                      (palm_x + palm_w * 0.45, knuckle_y + 4), 2)

    # --- Fingers (index=1 .. pinky=4), drawn from pinky to index so the active one overlaps correctly ---
    finger_base_y = knuckle_y

    finger_tip_positions = {}
    for i in [4, 3, 1, 2]:  # pinky, ring, index, middle — keeps middle on top
        offset = finger_offsets[i]
        base = (palm_x + offset, finger_base_y)
        is_active = (i == finger_idx)

        length = FINGER_LENGTHS[i] * 105
        # Slight natural spread angle outward from center, active finger straightens to reach the key
        rest_angle = math.radians(offset * 0.12)
        angle = 0.0 if is_active else rest_angle
        bend = 0.0 if is_active else 0.35  # resting fingers curl slightly; active one presses flat/straight

        base_w = 11 if i != 2 else 12
        tip_w = 6

        tip = _draw_finger(
            screen, base, angle, length, base_w, tip_w,
            FINGER_COLORS[i], bend, highlight=is_active,
        )
        finger_tip_positions[i] = tip

        if is_active and target_x is not None:
            # Snap the active fingertip visually onto the key target for a convincing "press"
            pygame.draw.line(screen, SKIN_SHADOW, tip, (target_x, target_y), 1)

    # --- Thumb: separate, shorter, angled outward from a FIXED side of the palm ---
    thumb_base = (palm_x + thumb_side * palm_w * 0.48, palm_y + 5)
    thumb_angle = math.radians(thumb_side * 55)
    is_thumb_active = (finger_idx == 0)
    thumb_length = FINGER_LENGTHS[0] * 105
    _draw_finger(
        screen, thumb_base, thumb_angle, thumb_length, 13, 7,
        FINGER_COLORS[0], 0.15, highlight=is_thumb_active,
    )

    # --- Wrist stub ---
    wrist_poly = [
        (palm_x - palm_w * 0.30, wrist_y),
        (palm_x + palm_w * 0.30, wrist_y),
        (palm_x + palm_w * 0.22, wrist_y + 40),
        (palm_x - palm_w * 0.22, wrist_y + 40),
    ]
    pygame.draw.polygon(screen, SKIN, wrist_poly)
    pygame.draw.polygon(screen, OUTLINE, wrist_poly, 2)