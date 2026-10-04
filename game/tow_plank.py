"""
Tug-of-War Beam Clash - two-player anime energy-ray battle (pygame).

POWER SOURCE: each player's beam power is their LIVE PLANK SCORE (0-100)
computed by the Flask plank server, which receives pose data from the two
Android phones running the plank-tracker app.

  Player 1 (left):  server slot 1 (first phone to connect) - gnome1.png (RED)
  Player 2 (right): server slot 2 (second phone to connect) - gnome2.png (BLUE)
  R: reset the match

WIN MECHANIC (hidden tug-of-war score):
  A hidden score starts at 0. Every frame (once the countdown finished):
      hidden += (p1.score - p2.score) / 100 * HIDDEN_RATE * dt
  The clash point between the beams follows this hidden score, and a
  player wins when it reaches +WIN_SCORE (P1) or -WIN_SCORE (P2).
  A lead must be HELD over time - momentary spikes don't win matches.
  Defaults are sized so a 10-point score gap (55 vs 45) held constantly
  wins in ~30 seconds; a 50-point gap wins in ~6 s; 100-0 blowout ~3 s.

Run on the same PC as the Flask server:
    pip install pygame
    python game.py
"""

import json
import math
import random
import threading
import urllib.request

import pygame

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 720
FPS = 60

GROUND_Y = 560          # where fighters stand
CLASH_ZONE_HALF = 0.30  # clash point can drift within middle 60% of screen

# Beam origin: fires from the gnome's UPPER HALF (gnomes are ~300 px tall),
# and slightly toward the center of the body so the beam leaves the torso
# instead of the outer edge.
BEAM_Y = GROUND_Y - 220        # beam height (upper half of the gnome)
BEAM_START_INSET = 80          # beam starts this far toward the center

SKY_TOP = (6, 10, 34)
SKY_MID = (22, 26, 70)
SKY_BOTTOM = (30, 60, 96)
GROUND_TOP = (28, 28, 54)
GROUND_LINE = (140, 220, 255)
MOON_COLOR = (230, 240, 255)
MOON_GLOW = (150, 190, 255)
AURORA_1 = (90, 255, 200)
AURORA_2 = (140, 120, 255)
RUNE_COLOR = (160, 230, 255)
STONE_DARK = (22, 22, 44)
STONE_MID = (36, 36, 66)
STONE_LIGHT = (52, 52, 90)

P1_COLOR = (255, 90, 30)     # red fighter (left, slot 1, gnome1.png)
P2_COLOR = (0, 200, 255)     # blue fighter (right, slot 2, gnome2.png)
P1_BEAM = (255, 120, 60)
P2_BEAM = (80, 200, 255)

HUD_BG = (16, 12, 40)
HUD_EDGE = (255, 255, 255)
HUD_TEXT = (255, 255, 255)
HUD_GOLD = (232, 195, 110)
HUD_GOLD_DARK = (150, 115, 55)
HUD_STONE = (34, 30, 62)

SCORE_MAX = 100.0
CLASH_EASE = 2.5            # how fast the visible clash point tracks the tug
START_DELAY = 10.0          # countdown seconds before the clash starts

# --- Hidden tug-of-war score (the real win mechanic) ---
# Every frame the hidden score moves by the CURRENT SCORE DIFFERENCE:
#     hidden += (p1.score - p2.score) / SCORE_MAX * HIDDEN_RATE * dt
#   hidden > 0  -> clash pushed toward P2's side (P1 winning)
#   hidden < 0  -> clash pushed toward P1's side (P2 winning)
# A player wins when hidden reaches +WIN_SCORE (P1) or -WIN_SCORE (P2),
# so a lead must be HELD over time, not just existed once.
#
# Sizing guide: time-to-win = WIN_SCORE / (score_gap% * HIDDEN_RATE)
#   e.g. HIDDEN_RATE=333, WIN_SCORE=1000, a 10-point gap (55 vs 45) held
#   constantly wins in 1000 / (0.1 * 333) ~= 30 seconds.
#   A 50-point gap (75 vs 25) wins in ~6 s; a full 100-point blowout ~3 s.
HIDDEN_RATE = 333.0         # tug points per second at max score gap
WIN_SCORE = 1000.0          # |hidden| needed to win

MIN_BEAM_POWER = 0.12       # beams never fully vanish - at 0 score they still
                            # trickle out at 12% strength (thin + dim).

# --- Player sprites ---
GNOME1_FILE = "gnome1.png"  # Player 1 (left, red)
GNOME2_FILE = "gnome2.png"  # Player 2 (right, blue)
GNOME_HEIGHT = 300           # sprite height in px (gnomes are 2x bigger)
# If a sprite file is missing, the stickman fallback is drawn instead.
# flip = True mirrors the sprite horizontally. gnome2 is flipped so both
# fighters face the middle; change these if your art already faces inward.
GNOME1_FLIP = False
GNOME2_FLIP = True

# --- Plank server integration ---
SERVER = "http://127.0.0.1:5000"   # Flask server (same PC)
# The server assigns player slots automatically: the FIRST phone that
# connects (posts frames) becomes P1, the second becomes P2. The game
# just asks the server for "whoever is in slot 1 / slot 2".
POLL_INTERVAL = 0.5                # seconds between score polls
SCORE_SMOOTH = 0.15                # per-frame lerp toward polled score
SCORE_FALL = 12.0                  # fallback drain/sec when a phone is offline
LOBBY_ONLINE_GRACE = 2.0           # both players online this long -> start


# --------------------------------------------------------------------------
# Plank score poller (background thread, talks to the Flask server)
# --------------------------------------------------------------------------
class PlankScorePoller:
    """Polls /score?slot=N for one player, latest score in .score.

    .online flips True on the first successful poll (device holds the
    slot), and only flips False when polls start failing (device gone
    or server down). Zero scores still count as online.
    """

    def __init__(self, slot):
        self.slot = slot
        self.score = 0.0          # last successfully polled plank score
        self.online = False      # True if the last poll succeeded
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        url = f"{SERVER}/score?slot={self.slot}"
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode())
                if "score" in data:
                    self.score = float(data["score"])
                    self.online = True
                else:
                    self.online = False   # no usable frames yet
            except Exception:
                self.online = False       # slot empty / server down
            self._stop.wait(POLL_INTERVAL)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
# Sprite loading
# --------------------------------------------------------------------------
def load_gnome(path, height, flip):
    """Load a gnome sprite, scaled to the given height.

    Returns None if the file is missing/broken (stickman fallback is
    used instead). Must be called AFTER pygame.display.set_mode because
    of convert_alpha().
    """
    try:
        img = pygame.image.load(path).convert_alpha()
    except (pygame.error, FileNotFoundError):
        print(f"[sprites] could not load {path}, using stickman fallback")
        return None
    w = max(8, int(img.get_width() * height / img.get_height()))
    img = pygame.transform.smoothscale(img, (w, height))
    if flip:
        img = pygame.transform.flip(img, True, False)
    return img


# --------------------------------------------------------------------------
# Fighter
# --------------------------------------------------------------------------
class Fighter:
    """One player: power (0-100) comes from their live plank score."""

    def __init__(self, side, color, beam_color, sprite=None):
        self.side = side                  # -1 left (P1), +1 right (P2)
        self.color = color
        self.beam_color = beam_color
        self.sprite = sprite              # gnome image, or None for stickman
        self.score = 0.0                 # smoothed power shown in-game
        self.target_score = 0.0          # latest polled plank score
        self.punch_t = 0.0               # animation timer for "charging" pose
        self.x = WIDTH // 2 + self.side * (WIDTH // 3)
        self.charge_flash = 0.0

    def update(self, dt):
        # Smoothly approach the polled plank score (removes poll jitter).
        k = min(1.0, SCORE_SMOOTH * FPS * dt)
        prev = self.score
        self.score += (self.target_score - self.score) * k

        # Charging flare follows the power level itself: strong planks
        # make the fighter glow; rising power gives a burst of flash.
        self.charge_flash = max(self.charge_flash,
                                self.score / SCORE_MAX * 0.6)
        if self.score > prev + 0.4:       # power jumped up -> punch flare
            self.punch_t = 1.0
            self.charge_flash = 1.0
        self.punch_t = max(0.0, self.punch_t - dt * 4)
        self.charge_flash = max(0.0, self.charge_flash - dt * 3)

    def beam_length(self):
        """How far the beam extends - stronger score pushes further."""
        return max(MIN_BEAM_POWER, self.score / SCORE_MAX)


# --------------------------------------------------------------------------
# Fighter drawing (sprites with stickman fallback)
# --------------------------------------------------------------------------
def draw_fighter(surface, x, y, color, facing, arm_extend, flash, sprite):
    """Draw a fighter at (x, y=feet). Gnome sprite if available,
    stickman fallback otherwise."""
    if sprite is not None:
        x, y = int(x), int(y)
        # Charging glow behind the sprite (feedback for strong planks).
        if flash > 0.05:
            bw = sprite.get_width() // 2 + 14
            bh = sprite.get_height() // 2 + 6
            s = pygame.Surface((bw * 2, bh * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, int(80 * flash)), (bw, bh),
                               min(bw, bh))
            surface.blit(s, (x - bw, y - sprite.get_height() + bh))
        surface.blit(sprite, (x - sprite.get_width() // 2,
                              y - sprite.get_height()))
    else:
        draw_stickman(surface, x, y, color, facing, arm_extend, flash)


def draw_fighter_rotated(surface, x, y, color, facing, arm_extend, flash,
                         angle, scale, sprite):
    """Loser's sky launch: rotate + shrink the sprite (or stickman)."""
    if sprite is not None:
        w, h = sprite.get_size()
        new_h = max(12, int(h * scale))
        new_w = max(4, int(w * new_h / h))
        img = pygame.transform.smoothscale(sprite, (new_w, new_h))
        if angle:
            img = pygame.transform.rotate(img, angle)
        surface.blit(img, (int(x) - img.get_width() // 2,
                           int(y) - img.get_height() // 2))
    else:
        draw_stickman_rotated(surface, x, y, color, facing, arm_extend,
                              flash, angle, scale)


def draw_stickman_rotated(surface, x, y, color, facing, arm_extend, flash, angle, scale):
    """Draw a stickman rotated + scaled - used for the loser's sky launch."""
    pad = 120
    tmp = pygame.Surface((pad * 2, pad * 2), pygame.SRCALPHA)
    draw_stickman(tmp, pad, pad + 100, color, facing, arm_extend, flash)
    if scale != 1.0:
        w = max(8, int(pad * 2 * scale))
        tmp = pygame.transform.smoothscale(tmp, (w, w))
    if angle:
        tmp = pygame.transform.rotate(tmp, angle)
    surface.blit(tmp, (int(x) - tmp.get_width() // 2,
                       int(y) - tmp.get_height() // 2))


def draw_stickman(surface, x, y, color, facing, arm_extend, flash=0.0):
    """Draw a stickman. facing: -1 = faces right, +1 = faces left."""
    y = int(y)
    c = color
    glow = [min(255, v + flash * 120) for v in color]

    head_r = 16
    pygame.draw.circle(surface, c, (x, y - 140), head_r, 3)
    pygame.draw.line(surface, c, (x, y - 124), (x, y - 70), 4)
    pygame.draw.line(surface, c, (x, y - 70), (x - facing * 26, y), 4)
    pygame.draw.line(surface, c, (x, y - 70), (x + facing * 20, y), 4)

    shoulder = (x, y - 108)
    reach = 34 + arm_extend * 30
    for dy in (-6, 6):
        hand = (x + facing * reach, y - 108 + dy)
        pygame.draw.line(surface, glow if flash > 0.1 else c, shoulder, hand, 4)
        pygame.draw.circle(surface, glow, hand, 5 if flash > 0.1 else 4)

    if flash > 0.05:
        for i in range(5):
            ly = y - 130 + i * 22
            lx = x - facing * (60 + i * 26)
            ln = int(30 + flash * 50 + i * 8)
            a = int(90 * flash)
            line = pygame.Surface((ln + 2, 3), pygame.SRCALPHA)
            pygame.draw.line(line, (*glow, a), (0, 1), (ln, 1), 2)
            surface.blit(line, (lx if facing > 0 else lx - ln, ly))

    if flash > 0.05:
        r = int(30 + flash * 26)
        aura = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(aura, (*glow, int(70 * flash)), (r, r), r)
        surface.blit(aura, (x - r, y - 105 - r))


# --------------------------------------------------------------------------
# Beams / clash
# --------------------------------------------------------------------------
def hidden_ratio(hidden):
    """Map the hidden tug score to the clash position: -1..+1.

    hidden = +WIN_SCORE -> +1 (P2's edge, P1 about to win)
    hidden = -WIN_SCORE -> -1 (P1's edge, P2 about to win)
    """
    return max(-1.0, min(1.0, hidden / WIN_SCORE))


def draw_beam(surface, p1, p2, ratio, t, particles):
    center_x = WIDTH // 2
    clash_x = center_x + int(ratio * WIDTH * CLASH_ZONE_HALF)
    beam_y = BEAM_Y

    def draw_half(start_x, end_x, color, core, power):
        if abs(end_x - start_x) < 2:
            return
        wob = lambda bx: math.sin(bx * 0.06 + t * 25) * (2 + power * 4)
        base_w = 3 + power * 12
        layers = (
            (base_w * 2.4, int(30 + 40 * power)),
            (base_w * 1.5, int(70 + 60 * power)),
            (base_w * 0.9, int(150 + 90 * power)),
        )
        pts = [(start_x, beam_y)] + [
            (bx, beam_y + wob(bx))
            for bx in range(int(start_x), int(end_x), 8)
        ] + [(end_x, beam_y + wob(end_x))]
        for half_w, a in layers:
            top = [(px, py - half_w) for px, py in pts]
            bot = [(px, py + half_w) for px, py in reversed(pts)]
            layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(layer, (*color, a), top + bot)
            surface.blit(layer, (0, 0))
        core_w = max(3, int(3 + power * 6 + math.sin(t * 30) * power * 2))
        pygame.draw.line(surface, core, (start_x, beam_y), (end_x, beam_y), core_w)

    draw_half(p1.x + BEAM_START_INSET, clash_x, P1_BEAM, (255, 255, 255), p1.beam_length())
    draw_half(p2.x - BEAM_START_INSET, clash_x, P2_BEAM, (255, 255, 255), p2.beam_length())

    power = (p1.score + p2.score) / SCORE_MAX
    r = int(20 + power * 44 + math.sin(t * 40) * 7)
    for i in range(3):
        rr = max(4, r - i * 7)
        s = pygame.Surface((rr * 2, rr * 2), pygame.SRCALPHA)
        col = (255, 255, 255) if i == 0 else ((150, 220, 255) if i == 1 else (255, 150, 90))
        pygame.draw.circle(s, (*col, 200 - i * 55), (rr, rr), rr)
        surface.blit(s, (clash_x - rr, beam_y - rr))
    ring_phase = (t * (1.5 + power)) % 1.0
    for k in range(2):
        ph = (ring_phase + k * 0.5) % 1.0
        rr = int(10 + ph * (60 + power * 120))
        alpha = int(180 * (1 - ph))
        if alpha > 5:
            ring = pygame.Surface((rr * 2, rr * 2), pygame.SRCALPHA)
            col = P1_BEAM if p1.score >= p2.score else P2_BEAM
            pygame.draw.circle(ring, (*col, alpha), (rr, rr), rr, max(2, rr // 8))
            surface.blit(ring, (clash_x - rr, beam_y - rr))
    for _ in range(int(1 + power * 3)):
        if random.random() < max(0.15, power * 0.95):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(2, 7 + power * 6)
            particles.append([clash_x, beam_y, math.cos(ang) * spd,
                              math.sin(ang) * spd - 2,
                              random.choice([P1_BEAM, P2_BEAM, (255, 255, 255)]),
                              random.uniform(0.3, 0.8),
                              1 + int(power * 3)])

    return clash_x


def draw_particles(surface, particles, dt):
    alive = []
    for p in particles:
        p[0] += p[2]
        p[1] += p[3]
        p[3] += 0.15
        p[5] -= dt
        if p[5] > 0:
            size = p[6] if len(p) > 6 else 2
            fade = min(1.0, p[5] * 3)
            r = max(1, int(size * fade))
            s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p[4], int(255 * fade)), (r + 1, r + 1), r)
            surface.blit(s, (int(p[0]) - r, int(p[1]) - r))
            alive.append(p)
    particles[:] = alive


# --------------------------------------------------------------------------
# Fantasy anime background & rune-circle ground (all static, rendered once)
# --------------------------------------------------------------------------
def glow_circle(surface, color, radius, alpha):
    """Soft radial glow of the given color."""
    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for i in range(24, 0, -1):
        r = int(radius * i / 24)
        a = int(alpha * (24 - i) / 24)
        pygame.draw.circle(s, (*color, a), (radius, radius), r)
    return s


def make_background():
    """Pre-render the full fantasy scene once. Returns a single Surface."""
    bg = pygame.Surface((WIDTH, HEIGHT))
    bg.fill(SKY_TOP)

    horizon = GROUND_Y

    for y in range(horizon):
        f = y / horizon
        if f < 0.65:
            a = f / 0.65
            col = tuple(int(SKY_TOP[i] + (SKY_MID[i] - SKY_TOP[i]) * a)
                        for i in range(3))
        else:
            a = (f - 0.65) / 0.35
            col = tuple(int(SKY_MID[i] + (SKY_BOTTOM[i] - SKY_MID[i]) * a)
                        for i in range(3))
        pygame.draw.line(bg, col, (0, y), (WIDTH, y))

    random.seed(42)
    for _ in range(120):
        x, y = random.randint(0, WIDTH), random.randint(0, horizon - 60)
        r = random.choice((1, 1, 1, 2, 2, 3))
        shade = random.randint(160, 255)
        if r == 3:
            c = (shade, shade, 255)
            pygame.draw.line(bg, c, (x - 6, y), (x + 6, y), 2)
            pygame.draw.line(bg, c, (x, y - 6), (x, y + 6), 2)
        else:
            bg.fill((shade, shade, min(255, shade + 15)), (x, y, r, r))

    aurora = pygame.Surface((WIDTH, horizon), pygame.SRCALPHA)
    for i, (base_col, y0, amp) in enumerate(((AURORA_1, 90, 40),
                                              (AURORA_2, 150, 55),
                                              (AURORA_1, 60, 30))):
        for x in range(0, WIDTH, 6):
            y = y0 + int(math.sin(x * 0.008 + i * 2.1) * amp)
            a = int(26 + 14 * math.sin(x * 0.004 + i))
            pygame.draw.line(aurora, (*base_col, a),
                             (x, y), (x, y + random.choice((4, 6, 9))), 5)
    bg.blit(aurora, (0, 0))

    mx, my, mr = int(WIDTH * 0.79), 140, 66
    bg.blit(glow_circle(bg, MOON_GLOW, mr * 4, 10), (mx - mr * 4, my - mr * 4))
    pygame.draw.circle(bg, MOON_COLOR, (mx, my), mr)
    for (cx, cy, cr) in ((-20, -14, 10), (16, 18, 8), (-2, 28, 5), (24, -20, 6)):
        pygame.draw.circle(bg, (205, 216, 240), (mx + cx, my + cy), cr)
    mx2, my2, mr2 = int(WIDTH * 0.68), 90, 22
    pygame.draw.circle(bg, (170, 200, 255), (mx2, my2), mr2)
    pygame.draw.circle(bg, (140, 175, 240), (mx2 - 6, my2 - 4), 4)

    random.seed(11)
    for (ix, iy, scale) in ((140, 210, 1.0), (1130, 260, 0.7),
                            (420, 150, 0.5), (960, 170, 0.45)):
        iw, ih = int(120 * scale), int(34 * scale)
        island = pygame.Surface((iw * 2, ih * 3), pygame.SRCALPHA)
        ox, oy = iw, 0
        pygame.draw.polygon(island, (40, 55, 70),
                            [(ox - iw, oy + ih), (ox + iw, oy + ih),
                             (ox + iw * 0.6, oy + ih + 6), (ox - iw * 0.6, oy + ih + 6)])
        pygame.draw.polygon(island, STONE_DARK,
                            [(ox - iw, oy + ih + 6), (ox + iw, oy + ih + 6),
                             (ox, oy + ih * 3)])
        cx, cy = ox + random.randint(-iw // 3, iw // 3), oy + ih
        cw, ch = int(16 * scale), int(34 * scale)
        crystal = pygame.Surface((cw * 2 + 20, ch * 2 + 20), pygame.SRCALPHA)
        gx = cw + 10
        gy = ch * 2 + 10
        crystal.blit(glow_circle(crystal, RUNE_COLOR, cw * 2, 60),
                     (gx - cw * 2, gy - ch - cw * 2))
        pygame.draw.polygon(crystal, (*RUNE_COLOR, 235),
                            [(gx, gy - ch * 2), (gx + cw, gy - ch),
                             (gx, gy), (gx - cw, gy - ch)])
        island.blit(crystal, (cx - gx, cy - gy + 4))
        for k in range(3):
            sx = ox + random.randint(-iw // 2, iw // 2)
            sy = oy + ih * 2 + random.randint(6, 26)
            pygame.draw.circle(island, (*STONE_MID, 220), (sx, sy), int(4 * scale))
        bg.blit(island, (ix - iw, iy))
    random.seed()

    random.seed(5)
    x = -40
    while x < WIDTH + 40:
        w = random.randint(120, 260)
        h = random.randint(60, 150)
        pts = [(x, horizon), (x + w // 2, horizon - h), (x + w, horizon)]
        pygame.draw.polygon(bg, (26, 30, 58), pts)
        x += w - 60
    random.seed()

    def draw_ruin(x, flip=False, broken=True):
        pill = []
        for i in range(3):
            px = x + (i * 70 if not flip else -i * 70)
            ph = 120 - (18 * i if broken else 0)
            top = horizon - ph
            pygame.draw.rect(bg, STONE_MID, (px, top, 26, ph))
            pygame.draw.rect(bg, STONE_LIGHT, (px, top, 26, 8))
            pygame.draw.rect(bg, STONE_LIGHT, (px, horizon - 10, 26, 10))
            pill.append((px + 13, top))
        for (ax, ay), (bx2, by2) in zip(pill, pill[1:]):
            seg = pygame.Surface((90, 26), pygame.SRCALPHA)
            pygame.draw.lines(seg, (*STONE_LIGHT, 235), False,
                              [(0, 24), (22, 8), (52, 4)], 8)
            if not flip:
                bg.blit(seg, (ax - 13, ay - 26))
            else:
                seg2 = pygame.transform.flip(seg, True, False)
                bg.blit(seg2, (bx2 - 52, by2 - 26))

    draw_ruin(60)
    draw_ruin(WIDTH - 210, flip=True)
    random.seed(9)
    for _ in range(40):
        rx = random.choice(list(range(50, 250)) + list(range(WIDTH - 250, WIDTH - 40)))
        ry = random.randint(horizon - 130, horizon - 8)
        bg.fill((60, 160, 120, ), (rx, ry, 2, 2))
    random.seed()

    pygame.draw.rect(bg, GROUND_TOP, (0, horizon, WIDTH, HEIGHT - horizon))
    random.seed(3)
    for i in range(-10, 11):
        x_edge = WIDTH // 2 + i * 130
        pygame.draw.polygon(bg, STONE_MID,
                            [(x_edge, horizon), (x_edge + 40, horizon),
                             (x_edge + 190 - i * 30, HEIGHT), (x_edge + 60 - i * 30, HEIGHT)])
    random.seed()
    random.seed(13)
    for i in range(26):
        cx = random.randint(0, WIDTH)
        cy = horizon + random.randint(8, HEIGHT - horizon - 8)
        x2, y2 = cx, cy
        pygame.draw.line(bg, STONE_DARK, (cx, cy), (x2 + random.randint(-60, 60),
                                                    y2 + random.randint(-14, 14)), 2)
    random.seed()

    rcx, rcy = WIDTH // 2, horizon + (HEIGHT - horizon) // 2 + 8
    rune = pygame.Surface((WIDTH, HEIGHT - horizon), pygame.SRCALPHA)
    for rr, w in ((150, 5), (120, 2), (86, 3)):
        pygame.draw.circle(rune, (*RUNE_COLOR, 120), (rcx, rcy - horizon), rr, w)
    for k in range(12):
        ang = k * math.tau / 12
        nx = rcx + math.cos(ang) * 150
        ny = rcy - horizon + math.sin(ang) * 150
        pygame.draw.circle(rune, (*RUNE_COLOR, 200), (int(nx), int(ny)), 5)
    bg.blit(rune, (0, horizon))
    bg.blit(glow_circle(bg, RUNE_COLOR, 170, 8), (rcx - 170, rcy - 170))

    rim = pygame.Surface((WIDTH, 10), pygame.SRCALPHA)
    pygame.draw.line(rim, (*GROUND_LINE, 255), (0, 4), (WIDTH, 4), 3)
    pygame.draw.line(rim, (*GROUND_LINE, 70), (0, 8), (WIDTH, 8), 1)
    bg.blit(rim, (0, horizon - 4))

    vig = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for i in range(60):
        a = int(1.1 * i)
        pygame.draw.rect(vig, (0, 0, 0, a), (i * 8, i * 5,
                                            WIDTH - i * 16, HEIGHT - i * 10), 2)
    bg.blit(vig, (0, 0))

    return bg


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------
def draw_gem(surface, x, y, r, color):
    pygame.draw.circle(surface, color, (x, y), r)
    pygame.draw.circle(surface, tuple(min(255, c + 90) for c in color),
                       (x - r // 3, y - r // 3), max(1, r // 2))
    pygame.draw.circle(surface, HUD_GOLD, (x, y), r, 1)


def arcane_plaque(surface, rect, gem_color, skew=0, gem_side="left"):
    x, y, w, h = rect
    s = skew if skew >= 0 else -skew
    if skew >= 0:
        pts = [(x + s, y), (x + w, y), (x + w, y + h), (x, y + h)]
    else:
        pts = [(x, y), (x + w - s, y), (x + w, y + h), (x, y + h)]
    pygame.draw.polygon(surface, (8, 6, 22),
                        [(px + 2, py + 4) for px, py in pts])
    pygame.draw.polygon(surface, HUD_STONE, pts)
    pygame.draw.polygon(surface, HUD_GOLD_DARK, pts, 6)
    pygame.draw.polygon(surface, HUD_GOLD, pts, 2)
    for (px, py), dx, dy in ((pts[0], 1, 1), (pts[1], -1, 1),
                             (pts[2], -1, -1), (pts[3], 1, -1)):
        pygame.draw.arc(surface, HUD_GOLD,
                        (px - 8 if dx > 0 else px - 4,
                         py - 8 if dy > 0 else py - 4, 12, 12),
                        0, math.tau, 2)
    if gem_side == "left":
        draw_gem(surface, x + s + 14, y + h // 2, 7, gem_color)
    else:
        draw_gem(surface, x + w - s - 14, y + h // 2, 7, gem_color)


def draw_countdown_scroll(surface, big_font, countdown):
    draw_hanging_scroll(surface, big_font, str(math.ceil(countdown)),
                        (255, 250, 230))


def draw_hanging_scroll(surface, big_font, text, color):
    glow_txt = big_font.render(text, True, RUNE_COLOR)
    rendered = big_font.render(text, True, color)

    rod_h = 14
    sheet_h = 150
    sheet_w = max(160, rendered.get_width() + 70)
    rod_w = sheet_w + 30
    cx, cy = WIDTH // 2, HEIGHT // 2

    plate = pygame.Surface((sheet_w + 40, rod_h + 14 + sheet_h + 30),
                           pygame.SRCALPHA)
    px0 = (sheet_w + 40 - rod_w) // 2
    rod_y = 0
    sheet_x = (sheet_w + 40 - sheet_w) // 2
    sheet_y = rod_y + rod_h + 14

    for sx in (sheet_x, sheet_x + sheet_w):
        pygame.draw.line(plate, (*HUD_GOLD_DARK, 255),
                         (px0 + (0 if sx == sheet_x else rod_w),
                          rod_y + rod_h // 2),
                         (sx, sheet_y + 4), 3)

    pygame.draw.rect(plate, (*HUD_GOLD_DARK, 255), (px0, rod_y, rod_w, rod_h),
                     border_radius=7)
    pygame.draw.rect(plate, (*HUD_GOLD, 255), (px0 + 2, rod_y + 2,
                                              rod_w - 4, rod_h - 4),
                     border_radius=5)
    for ex in (px0 - 8, px0 + rod_w - 2):
        pygame.draw.circle(plate, (*HUD_GOLD_DARK, 255), (ex + 5, rod_y + rod_h // 2), 8)
        pygame.draw.circle(plate, (*HUD_GOLD, 255), (ex + 5, rod_y + rod_h // 2), 5)

    sheet = [(sheet_x, sheet_y),
            (sheet_x + sheet_w, sheet_y),
            (sheet_x + sheet_w - 12, sheet_y + sheet_h - 14),
            (sheet_x + sheet_w // 2, sheet_y + sheet_h),
            (sheet_x + 12, sheet_y + sheet_h - 14)]
    pygame.draw.polygon(plate, (30, 24, 56, 245), sheet)
    pygame.draw.polygon(plate, (*HUD_GOLD_DARK, 255), sheet, 6)
    pygame.draw.polygon(plate, (*HUD_GOLD, 255), sheet, 2)
    draw_gem(plate, px0 - 3, rod_y + rod_h // 2, 5, RUNE_COLOR)
    draw_gem(plate, px0 + rod_w + 3, rod_y + rod_h // 2, 5, RUNE_COLOR)

    surface.blit(plate, (cx - plate.get_width() // 2,
                         cy - (sheet_h // 2) - (sheet_y - 14)))

    text_cx = cx - rendered.get_width() // 2
    text_cy = cy - rendered.get_height() // 2
    glow = glow_circle(surface, RUNE_COLOR, 60, 5)
    surface.blit(glow, (cx - 60, cy - 60))
    surface.blit(glow_txt, (text_cx + 2, text_cy + 2))
    surface.blit(rendered, (text_cx, text_cy))


def draw_hud(surface, font, big_font, p1, p2, winner, countdown=0.0,
             p1_online=True, p2_online=True, lobby=False, hidden=0.0):
    for i, (p, col, side, online) in enumerate(((p1, P1_COLOR, 1, p1_online),
                                                (p2, P2_COLOR, -1, p2_online))):
        label = font.render(f"PLAYER {i + 1}" + ("" if online else " (offline)"),
                            True, HUD_GOLD if online else (200, 80, 80))
        score = font.render(f"{p.score:5.1f}", True, HUD_TEXT)
        pw, ph = 180, 58
        if side == 1:
            arcane_plaque(surface, (24, 16, pw, ph), col, skew=14)
            text_left = 24 + 14 + 14
            text_w = pw - 14 - 14 - 14
            surface.blit(label, (text_left + (text_w - label.get_width()) // 2,
                                 16 + 7))
            surface.blit(score, (text_left + (text_w - score.get_width()) // 2,
                                 16 + ph - 7 - score.get_height()))
        else:
            arcane_plaque(surface, (WIDTH - 24 - pw, 16, pw, ph), col,
                          skew=-14, gem_side="right")
            text_left = WIDTH - 24 - pw + 14
            text_w = pw - 14 - 14 - 14
            surface.blit(label, (text_left + (text_w - label.get_width()) // 2,
                                 16 + 7))
            surface.blit(score, (text_left + (text_w - score.get_width()) // 2,
                                 16 + ph - 7 - score.get_height()))

    bar_w, bar_h = 500, 18
    bx, by = (WIDTH - bar_w) // 2, 24
    for ex, flip in ((bx - 26, False), (bx + bar_w + 26, True)):
        cap = pygame.Surface((34, 46), pygame.SRCALPHA)
        pygame.draw.polygon(cap, (*HUD_STONE, 255),
                            [(2, 8), (30, 8), (30, 38), (2, 38)])
        pygame.draw.polygon(cap, (*HUD_GOLD_DARK, 255),
                            [(2, 8), (30, 8), (30, 38), (2, 38)], 5)
        pygame.draw.polygon(cap, (*HUD_GOLD, 255),
                            [(2, 8), (30, 8), (30, 38), (2, 38)], 2)
        draw_gem(cap, 16, 23, 6, RUNE_COLOR)
        if flip:
            cap = pygame.transform.flip(cap, True, False)
        surface.blit(cap, (ex - 17, by - 14))
    pygame.draw.polygon(surface, (8, 6, 22),
                        [(bx - 6, by - 2), (bx + bar_w + 6, by - 2),
                         (bx + bar_w + 6, by + bar_h + 4), (bx - 6, by + bar_h + 4)])
    pygame.draw.rect(surface, (24, 20, 52),
                     (bx, by, bar_w, bar_h), border_radius=4)
    pygame.draw.rect(surface, HUD_GOLD_DARK, (bx, by, bar_w, bar_h),
                     border_radius=4, width=4)
    pygame.draw.rect(surface, HUD_GOLD, (bx + 2, by + 2, bar_w - 4, bar_h - 4),
                     border_radius=3, width=1)
    fill = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
    pygame.draw.rect(fill, (*RUNE_COLOR, 26), (0, 0, bar_w, bar_h), border_radius=3)
    surface.blit(fill, (bx, by))
    pygame.draw.polygon(surface, P1_COLOR,
                        [(bx, by), (bx + 60, by), (bx + 44, by + bar_h), (bx, by + bar_h)])
    pygame.draw.polygon(surface, P2_COLOR,
                        [(bx + bar_w - 44, by), (bx + bar_w, by),
                         (bx + bar_w, by + bar_h), (bx + bar_w - 60, by + bar_h)])
    # The tug marker follows the hidden tug-of-war score.
    ratio = hidden_ratio(hidden)
    px = bx + bar_w // 2 + int(ratio * (bar_w // 2 - 4))
    mk = pygame.Surface((44, 52), pygame.SRCALPHA)
    pygame.draw.circle(mk, (255, 255, 255, 70), (22, 26), 21)
    pygame.draw.circle(mk, (*HUD_GOLD, 255), (22, 26), 13, 2)
    pygame.draw.circle(mk, (255, 255, 255, 255), (22, 26), 8)
    pygame.draw.line(mk, (17, 17, 34, 255), (22, 15), (22, 37), 2)
    surface.blit(mk, (px - 22, by + bar_h // 2 - 26))
    pygame.draw.line(surface, (255, 255, 255), (px, by - 4), (px, by + bar_h + 4), 4)

    msg = "HOLD YOUR PLANK!  Better form = stronger beam    R: reset"
    if lobby:
        msg = (f"P1: {'CONNECTED' if p1_online else 'waiting...'}   "
               f"P2: {'CONNECTED' if p2_online else 'waiting...'}")
    # Lobby banner: waiting for players to connect. Note that .online for
    # a poller turns True as soon as the server has that slot assigned, so
    # this is really "both slots filled".
    if lobby:
        draw_hanging_scroll(surface, big_font, "WAITING FOR PLAYERS...",
                            (255, 240, 200))
    elif countdown > 0:
        draw_countdown_scroll(surface, big_font, countdown)
        msg = "GET READY...  assume plank positions!"
    if winner:
        draw_hanging_scroll(surface, big_font, f"PLAYER {winner} WINS!",
                            P1_COLOR if winner == 1 else P2_COLOR)
        msg = "Press R for a rematch"

    hh = 36
    hy = HEIGHT - hh
    hint_panel = pygame.Surface((WIDTH, hh), pygame.SRCALPHA)
    hint_panel.fill((16, 12, 40, 160))
    pygame.draw.line(hint_panel, (*HUD_GOLD_DARK, 200), (0, 1), (WIDTH, 1), 2)
    pygame.draw.line(hint_panel, (*HUD_GOLD_DARK, 200), (0, hh - 2), (WIDTH, hh - 2), 2)
    surface.blit(hint_panel, (0, hy))
    hint = font.render(msg, True, HUD_GOLD)
    surface.blit(hint, ((WIDTH - hint.get_width()) // 2,
                        hy + (hh - hint.get_height()) // 2))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Plank-Powered Beam Clash")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,menlo,monospace", 24)
    big_font = pygame.font.SysFont("impact,arialblack,arial", 72)

    background = make_background()

    # Load player sprites (must happen after set_mode because of
    # convert_alpha()). Falls back to stickmen if the files are missing.
    gnome1 = load_gnome(GNOME1_FILE, GNOME_HEIGHT, GNOME1_FLIP)
    gnome2 = load_gnome(GNOME2_FILE, GNOME_HEIGHT, GNOME2_FLIP)

    p1 = Fighter(-1, P1_COLOR, P1_BEAM, sprite=gnome1)
    p2 = Fighter(+1, P2_COLOR, P2_BEAM, sprite=gnome2)

    # Score pollers: each fighter is fed by one phone's plank score.
    # Slots are assigned by the server: first phone to connect = P1.
    poll1 = PlankScorePoller(slot=1)
    poll2 = PlankScorePoller(slot=2)

    particles = []
    winner = 0
    loser_fly_t = -1.0
    loser = None
    t = 0.0
    countdown = 0.0               # 0 means "waiting in lobby"
    lobby_ok_t = 0.0              # how long BOTH players have been online
    hidden = 0.0                  # hidden tug-of-war score (the real win state)
    display_ratio = 0.0
    shake_level = 0.0             # smoothed screen-shake intensity (0..5)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        t += dt

        # --- Lobby: wait for BOTH phones to connect before countdown ---
        if countdown <= 0 and not winner and lobby_ok_t < LOBBY_ONLINE_GRACE:
            if poll1.online and poll2.online:
                lobby_ok_t += dt
            else:
                lobby_ok_t = 0.0
            if lobby_ok_t >= LOBBY_ONLINE_GRACE:
                countdown = START_DELAY   # both connected -> start countdown
        # Countdown only runs once the lobby has been cleared (both
        # players were online); no tug progress while it is in progress.
        if countdown > 0 and lobby_ok_t >= LOBBY_ONLINE_GRACE:
            countdown -= dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    p1.score = p2.score = 0.0
                    p1.target_score = p2.target_score = 0.0
                    particles.clear()
                    winner = 0
                    hidden = 0.0
                    countdown = 0.0             # back to lobby
                    lobby_ok_t = 0.0
                    display_ratio = 0.0
                    loser_fly_t = -1.0
                    loser = None

        # Feed polled plank scores into the fighters.
        # Offline phones simply stop feeding (score decays below).
        if poll1.online:
            p1.target_score = poll1.score
        else:
            p1.target_score = max(0.0, p1.target_score - SCORE_FALL * dt)
        if poll2.online:
            p2.target_score = poll2.score
        else:
            p2.target_score = max(0.0, p2.target_score - SCORE_FALL * dt)

        p1.update(dt)
        p2.update(dt)

        # --- Hidden tug-of-war score: integrate the score difference ---
        # Only accumulates once the countdown has finished (never during
        # the lobby or the loading countdown). A momentary lead does
        # nothing - the advantage must be held over time.
        if not winner and countdown <= 0 and lobby_ok_t >= LOBBY_ONLINE_GRACE:
            diff = (p1.score - p2.score) / SCORE_MAX
            hidden += diff * HIDDEN_RATE * dt
            hidden = max(-WIN_SCORE * 1.5, min(WIN_SCORE * 1.5, hidden))

        # The clash point (beam meeting position) follows the hidden score.
        target = hidden_ratio(hidden)
        display_ratio += (target - display_ratio) * min(1.0, CLASH_EASE * dt)
        if countdown > 0 or lobby_ok_t < LOBBY_ONLINE_GRACE:
            display_ratio = 0.0  # beams locked at center until match starts

        if loser is not None and loser_fly_t >= 0.0:
            loser_fly_t += dt

        ratio = display_ratio
        clash_x = WIDTH // 2 + int(ratio * WIDTH * CLASH_ZONE_HALF)

        # Screen shake is the effect of the beams CLASHING - while the
        # match is live it follows total beam power, but once a player
        # has won the clash is over, so it eases smoothly to zero.
        if winner or (loser is not None and loser_fly_t >= 0.0):
            shake_level = max(0.0, shake_level - dt * 3.0)
        else:
            shake_level += (((p1.score + p2.score) / SCORE_MAX * 5)
                            - shake_level) * min(1.0, 8.0 * dt)
        shake = int(round(shake_level))
        offset = (random.randint(-shake, shake), random.randint(-shake, shake))

        world = background.copy()
        if not winner:
            clash_x = draw_beam(world, p1, p2, ratio, t, particles) or clash_x
        draw_particles(world, particles, dt)
        loser_flying = (loser is not None and loser_fly_t >= 0.0)
        if not (loser_flying and loser is p1):
            draw_fighter(world, p1.x, GROUND_Y, p1.color, +1,
                         p1.punch_t, p1.charge_flash, p1.sprite)
        if not (loser_flying and loser is p2):
            draw_fighter(world, p2.x, GROUND_Y, p2.color, -1,
                         p2.punch_t, p2.charge_flash, p2.sprite)
        if loser_flying:
            ft = loser_fly_t
            winner_f = p2 if loser is p1 else p1
            BLAST_AT = 0.06
            hand_x = winner_f.x + (BEAM_START_INSET if winner_f is p1 else -BEAM_START_INSET)
            hand_y = BEAM_Y
            pow_x = loser.x - loser.side * BEAM_START_INSET
            power = winner_f.score / SCORE_MAX
            base_w = 3 + power * 12
            vb_t = min(1.0, ft / BLAST_AT)
            if vb_t < 1.0:
                start_x = hand_x + (pow_x - hand_x) * (vb_t ** 1.3)
                wob = lambda bx: math.sin(bx * 0.05 + t * 30) * (2 + power * 4)
                n = 14
                pts = [(start_x, hand_y)] + [
                    (start_x + (pow_x - start_x) * (i / n),
                     hand_y + wob(i))
                    for i in range(1, n)] + [(pow_x, hand_y)]
                for half_w, a in ((base_w * 2.4, int(30 + 40 * power)),
                                   (base_w * 1.5, int(70 + 60 * power)),
                                   (base_w * 0.9, int(150 + 90 * power))):
                    layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    top = [(px, py - half_w) for px, py in pts]
                    bot = [(px, py + half_w) for px, py in reversed(pts)]
                    pygame.draw.polygon(layer, (*winner_f.beam_color, a), top + bot)
                    world.blit(layer, (0, 0))
                core_w = max(3, int(3 + power * 6))
                pygame.draw.line(world, (255, 255, 255),
                                 (start_x, hand_y), (pow_x, hand_y), core_w)
                world.blit(glow_circle(world, (255, 255, 255), 16, 3),
                           (start_x - 16, hand_y - 16))
            elif ft < BLAST_AT + 0.20:
                pw = 1.0 - (ft - BLAST_AT) / 0.20
                for _ in range(9):
                    a2 = random.uniform(0, math.tau)
                    spd = random.uniform(5, 15)
                    particles.append([pow_x, hand_y,
                                      math.cos(a2) * spd,
                                      math.sin(a2) * spd - 1,
                                      random.choice([(255, 255, 255),
                                                     winner_f.beam_color,
                                                     winner_f.beam_color]),
                                      random.uniform(0.25, 0.55), 3])
                pr = int(30 + (1 - pw) * 170)
                pa = int(200 * pw)
                if pa > 5:
                    ring = pygame.Surface((pr * 2, pr * 2), pygame.SRCALPHA)
                    pygame.draw.circle(ring, (*winner_f.beam_color, pa),
                                       (pr, pr), pr, max(2, pr // 9))
                    world.blit(ring, (pow_x - pr, hand_y - pr))
                    ring2 = pygame.Surface((pr * 2, pr * 2), pygame.SRCALPHA)
                    pygame.draw.circle(ring2, (255, 255, 255, pa),
                                       (pr, pr), max(2, pr * 2 // 3), 2)
                    world.blit(ring2, (pow_x - pr, hand_y - pr))
                pygame.draw.circle(world, (255, 255, 255),
                                   (int(pow_x), int(hand_y)), max(3, int(26 * pw)), 0)
                if ft - dt < BLAST_AT:
                    for _ in range(26):
                        a2 = random.uniform(0, math.tau)
                        spd = random.uniform(4, 13)
                        particles.append([pow_x, hand_y,
                                          math.cos(a2) * spd,
                                          math.sin(a2) * spd - 2,
                                          random.choice([(255, 255, 255),
                                                         winner_f.beam_color]),
                                          random.uniform(0.4, 0.9), 3])
                    for _ in range(26):
                        ang = random.uniform(-math.pi, 0)
                        spd = random.uniform(2, 8)
                        particles.append([loser.x, GROUND_Y - 8,
                                          math.cos(ang) * spd * loser.side,
                                          math.sin(ang) * spd,
                                          random.choice([loser.beam_color,
                                                         (255, 255, 255)]),
                                          random.uniform(0.4, 0.9), 3])

            fly_t = max(0.0, ft - BLAST_AT)
            ease = min(1.0, fly_t / 2.0)
            prog = 1.0 - (1.0 - ease) ** 2
            corner_x = 150 if loser.side < 0 else WIDTH - 150
            corner_y = 130
            if fly_t <= 0.0:
                shake_x = random.randint(-2, 2)
                fx = loser.x + shake_x
                fy = GROUND_Y
            else:
                fx = loser.x + (corner_x - loser.x) * prog
                fy = (GROUND_Y - 40) + (corner_y - (GROUND_Y - 40)) * (prog ** 1.6)
            angle = fly_t * 540 * loser.side
            scale = max(0.25, 1.0 - prog * 0.75)

            vanish_t = max(0.0, ft - BLAST_AT - 2.0)
            if vanish_t < 0.35 and scale <= 0.26:
                fr = int(30 + vanish_t * 200)
                fa = int(255 * (1 - vanish_t / 0.35))
                flash = glow_circle(world, (255, 255, 255), fr + 30, fa // 4)
                world.blit(flash, (fx - fr - 30, fy - fr - 30))
                fl2 = glow_circle(world, loser.beam_color, fr, fa // 2)
                world.blit(fl2, (fx - fr, fy - fr))
                pygame.draw.circle(world, (255, 255, 255), (int(fx), int(fy)),
                                   max(2, int(fr * 0.25)), 0)
                if vanish_t <= dt:
                    for _ in range(18):
                        ang = random.uniform(0, math.tau)
                        spd = random.uniform(2, 6)
                        particles.append([fx, fy, math.cos(ang) * spd,
                                          math.sin(ang) * spd,
                                          random.choice([(255, 255, 255),
                                                          loser.beam_color]),
                                          random.uniform(0.4, 0.9), 3])
            if vanish_t < 0.35:
                if fly_t <= 0.0:
                    draw_fighter(world, fx, fy, loser.color, -loser.side,
                                 0.9, 0.6, loser.sprite)
                else:
                    draw_fighter_rotated(world, fx, fy, loser.color, -loser.side,
                                         0.9, 0.6, angle, scale, loser.sprite)
        draw_hud(world, font, big_font, p1, p2, winner, countdown,
                 poll1.online, poll2.online,
                 lobby=(countdown <= 0 and not winner
                        and lobby_ok_t < LOBBY_ONLINE_GRACE),
                 hidden=hidden)
        screen.fill((0, 0, 0))
        screen.blit(world, offset)
        pygame.display.flip()

        # Win by hidden tug score: +WIN_SCORE -> P1 wins, -WIN_SCORE -> P2.
        if (not winner and countdown <= 0
                and lobby_ok_t >= LOBBY_ONLINE_GRACE):
            if hidden >= WIN_SCORE:
                winner = 1
            elif hidden <= -WIN_SCORE:
                winner = 2
            if winner:
                loser = p1 if winner == 2 else p2
                loser_fly_t = 0.0

    poll1.stop()
    poll2.stop()
    pygame.quit()


if __name__ == "__main__":
    main()