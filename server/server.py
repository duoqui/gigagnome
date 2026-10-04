"""
Flask server that collects pose landmark frames from the Android app
and computes a plank quality score (0-100).

Run:
    pip install flask
    python app.py
    # serves on http://0.0.0.0:5000

Endpoints:
    POST /session/start          -> start a new scoring session
    POST /pose                   -> ingest one frame of landmarks (called by the app)
    GET  /score                  -> score the ingested frames of current session
    POST /session/start?reset=1  -> reset
"""

import math
import time
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)

# ------------------------------------------------------------
# ML Kit PoseLandmark type indices (sent by the Android app)
# ------------------------------------------------------------
NOSE = 0
MOUTH_LEFT = 9  # not used but reserved
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

MIN_LIKELIHOOD = 0.5  # ignore low-confidence points


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

# ------------------------------------------------------------
# Sessions — keyed by device_id so many phones can post
# concurrently, each with its own frame buffer and score.
# ------------------------------------------------------------

class Session:
    def __init__(self):
        self.device_id = "default"
        self.frames = []       # list of {t: float, points: {type: (x,y,lik)}}
        self.started_at = time.time()
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.frames = []
            self.started_at = time.time()


sessions = {}                 # device_id -> Session
sessions_lock = threading.Lock()
SESSION_IDLE_TIMEOUT = 300.0  # drop sessions unused for 5 min

# Player slots: the first device to connect becomes P1, the second P2.
# A device keeps its slot until its session is garbage-collected.
player_slots = {}             # slot number (1/2) -> device_id
slot_of_device = {}           # device_id -> slot number
slots_lock = threading.Lock()


def slot_for(device_id):
    """Assign a player slot: first connected device = 1, second = 2.

    Further devices share slot 2 (or you can extend this to reject them).
    Reuses the same slot on reconnect (device keeps its slot while its
    session is still alive).
    """
    with slots_lock:
        if device_id in slot_of_device:
            return slot_of_device[device_id]
        for slot in (1, 2):
            if slot not in player_slots:
                player_slots[slot] = device_id
                slot_of_device[device_id] = slot
                return slot
        # both slots taken -> default everyone else to slot 2
        slot_of_device[device_id] = 2
        return 2


def device_for_slot(slot):
    """The device_id currently holding a player slot (or None)."""
    with slots_lock:
        return player_slots.get(slot)


def get_session(device_id):
    """Return the session for a device, creating it on first use."""
    if not device_id or not isinstance(device_id, str):
        device_id = "default"
    with sessions_lock:
        _drop_stale_sessions_locked()
        if device_id not in sessions:
            sessions[device_id] = Session()
            sessions[device_id].device_id = device_id
            # First device to connect takes slot 1, second takes slot 2.
            slot_for(device_id)
        return sessions[device_id]


def _drop_stale_sessions_locked():
    now = time.time()
    stale = [d for d, s in sessions.items()
             if s.frames and now - s.frames[-1]["t"] > SESSION_IDLE_TIMEOUT]
    for d in stale:
        del sessions[d]
        # free the player slot this device was holding, if any
        slot = slot_of_device.pop(d, None)
        if slot is not None and player_slots.get(slot) == d:
            del player_slots[slot]


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------

def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle_at(a, b, c):
    """Angle (deg) at b formed by a-b-c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return 180.0
    cos = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos))


def signed_line_deviation(p, a, b):
    """
    Signed perpendicular distance of p from line a-b, normalized by |a-b|.
    Sign tells which side of the line the point is on (2D cross product).
    """
    ax, ay = a
    bx, by = b
    px, py = p
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    length = math.hypot(bx - ax, by - ay)
    if length == 0:
        return 0.0
    return cross / length  # in units of the shoulder->ankle length


# ------------------------------------------------------------
# Per-frame analysis
# ------------------------------------------------------------

def analyze_frame(pts):
    """
    pts: dict {type_index: (x, y, likelihood)}
    Returns dict with geometry metrics or None if not enough data.
    """
    required = [LEFT_SHOULDER, RIGHT_SHOULDER,
                LEFT_HIP, RIGHT_HIP,
                LEFT_ANKLE, RIGHT_ANKLE]
    if not all(t in pts for t in required):
        return None

    shoulder = midpoint(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER])
    hip = midpoint(pts[LEFT_HIP], pts[RIGHT_HIP])
    ankle = midpoint(pts[LEFT_ANKLE], pts[RIGHT_ANKLE])

    # --- Plank gating: is this pose actually a plank? ---
    # A plank (as seen side-on) means a roughly HORIZONTAL body:
    # shoulder, hip and ankle at similar heights, and the shoulder-ankle
    # span roughly horizontal. A standing person is also a "straight
    # line" but a VERTICAL one - without this check they'd score 100.
    body_dx = abs(ankle[0] - shoulder[0])
    body_dy = abs(ankle[1] - shoulder[1])
    body_len = math.hypot(body_dx, body_dy)
    if body_len == 0:
        return None
    horizontalness = body_dx / body_len        # 1 = lying flat, 0 = upright
    # Shoulder and ankle must be at similar heights (within 35% of
    # body length) and the body must be more horizontal than vertical.
    height_diff = max(abs(pts[LEFT_SHOULDER][1] - pts[LEFT_ANKLE][1]),
                      abs(pts[RIGHT_SHOULDER][1] - pts[RIGHT_ANKLE][1]))
    if horizontalness < 0.7 or height_diff > body_len * 0.35:
        return None                            # not a plank pose

    # --- Torso alignment: angle at hip (shoulder-hip-ankle) ---
    # 180 deg = perfectly straight line (correct plank).
    torso_angle = angle_at(shoulder, hip, ankle)

    # --- Hip offset: signed deviation of the hip from the
    #     shoulder->ankle straight line, normalized by line length.
    #     ~0    -> aligned (correct)
    #     > 0   -> hip on one side (hips up / crooked, HC)
    #     < 0   -> hip on other side (hips dropped, HD; or arched back, BC)
    hip_dev = signed_line_deviation(hip, shoulder, ankle)

    # --- Head position: angle at shoulder (hip-shoulder-nose). ---
    #     ~0 (nose roughly on the shoulder->hip extension) = neutral head.
    head_tilt = None
    if NOSE in pts:
        head_tilt = angle_at(hip, shoulder, pts[NOSE])
        # 0 = looking straight ahead (neutral). Large = tilted.
        head_tilt = head_tilt - 180.0 if head_tilt > 90 else head_tilt

    # --- Arm support check (optional): wrists should be near shoulders. ---
    arm_ok = True
    if LEFT_WRIST in pts and LEFT_ELBOW in pts:
        d = math.hypot(pts[LEFT_WRIST][0] - pts[LEFT_SHOULDER][0],
                      pts[LEFT_WRIST][1] - pts[LEFT_SHOULDER][1])
        body_len = math.hypot(shoulder[0] - ankle[0], shoulder[1] - ankle[1])
        if body_len > 0 and d / body_len > 0.9:
            arm_ok = False

    return {
        "torso_angle": torso_angle,
        "hip_dev": hip_dev,
        "head_tilt": head_tilt,
        "arm_ok": arm_ok,
    }


def classify(metrics):
    """
    Classify the frame into the technique categories from
    PMC9228676 (European Fitness Badge deviations):
      correct / head tilt fwd (HTFN) / head tilt back (HTBN) /
      back convex-arched (BC) / hips crooked-up (HC) / hips dropped (HD)
    """
    torso = metrics["torso_angle"]
    dev = metrics["hip_dev"]
    head = metrics["head_tilt"]

    # thresholds (tuned, adjust with real data)
    if torso < 150:
        if dev < 0:
            label = "hips_dropped (HD)"
        else:
            label = "back_arched (BC)"
    elif abs(dev) > 0.12:
        label = "hips_crooked (HC)" if dev > 0 else "hips_dropped (HD)"
    elif head is not None and abs(head) > 35:
        label = "head_tilted (HTFN/HTBN)"
    else:
        label = "correct"

    return label


def frame_score(metrics):
    """
    Score a single frame, 0-100, with weighted penalties:
      - torso straightness (shoulder-hip-ankle angle)  50 pts
      - hip alignment vs shoulder-ankle line          30 pts
      - head neutral position                          20 pts
    """
    score = 0.0

    # 1) Torso angle: 180 = perfect. Penalty grows with deviation.
    torso_dev = abs(180.0 - metrics["torso_angle"])
    torso_pts = max(0.0, 50.0 * (1.0 - torso_dev / 45.0))  # 45+ deg off = 0
    score += torso_pts

    # 2) Hip deviation from the line: 0 = perfect.
    hip_pts = max(0.0, 30.0 * (1.0 - abs(metrics["hip_dev"]) / 0.25))
    score += hip_pts

    # 3) Head tilt: within ~15 deg = full points, linear decay to 40 deg.
    if metrics["head_tilt"] is None:
        score += 10.0  # partial credit when nose not visible
    else:
        t = abs(metrics["head_tilt"])
        head_pts = max(0.0, 20.0 * (1.0 - max(0.0, t - 15.0) / 25.0))
        score += head_pts

    if not metrics["arm_ok"]:
        score *= 0.8  # weak penalty for odd hand placement

    return max(0.0, min(100.0, score))


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.post("/session/start")
def session_start():
    data = request.get_json(force=True, silent=True) or {}
    s = get_session(data.get("device_id") or request.args.get("device_id"))
    s.reset()
    return jsonify({"status": "session started", "device_id": request.args.get("device_id") or data.get("device_id", "default")})


@app.post("/pose")
def pose():
    data = request.get_json(force=True)
    s = get_session(data.get("device_id", "default"))
    pts = {}
    for p in data.get("points", []):
        if p.get("likelihood", 0) >= MIN_LIKELIHOOD:
            pts[p["type"]] = (p["x"], p["y"], p["likelihood"])
    with s.lock:
        s.frames.append({
            "t": time.time(),
            "pts": pts,
        })
        # keep last 30 s of frames to bound memory
        cutoff = time.time() - 30.0
        s.frames = [f for f in s.frames if f["t"] >= cutoff]
        n = len(s.frames)

    # Every 25 ingested frames (~5 s at the app's 5 frames/sec throttle),
    # compute and print the live score next to the access log.
    if n % 25 == 0:
        with s.lock:
            snapshot = list(s.frames)
        results = [analyze_frame(f["pts"]) or "NO_POSE" for f in snapshot]
        usable = [r for r in results if r != "NO_POSE"]
        if usable:
            scores = [frame_score(r) if r != "NO_POSE" else 0.0
                      for r in results]
            labels = [classify(r) if r != "NO_POSE" else "no_pose"
                      for r in results]
            avg = sum(scores) / len(scores)
            dominant = max(set(labels), key=labels.count)
            print(f"[score] device={s.device_id} "
                  f"frames={len(results)} score={avg:.0f}/100 ({dominant})")

    return jsonify({"status": "ok", "frames": n})


@app.get("/score")
def score():
    # Accepts ?device_id=... (as before) OR ?slot=1|2 (game-friendly).
    device_id = request.args.get("device_id")
    if device_id is None and "slot" in request.args:
        device_id = device_for_slot(int(request.args["slot"]))
    if device_id is None:
        return jsonify({"error": "no device in that slot"}), 404
    s = get_session(device_id)
    with s.lock:
        frames = list(s.frames)

    # Frames where the pose isn't plank-detectable score 0 (not skipped),
    # so standing up or leaving frame tanks the score instead of being
    # conveniently ignored by the average.
    results = [analyze_frame(f["pts"]) or "NO_POSE" for f in frames]

    usable = [r for r in results if r != "NO_POSE"]
    if not usable:
        # no usable frames: person never in a plank-detectable pose
        # (or nothing posted yet) -> score 0, not an error.
        return jsonify({
            "device_id": device_id,
            "slot": slot_of_device.get(device_id),
            "score": 0,
            "label": "no_pose" if frames else "no_frames",
            "frames_analyzed": 0,
            "frames_total": len(frames),
            "avg_metrics": None,
        })

    # score each frame: non-plank frames are hard zeros
    scores = [frame_score(r) if r != "NO_POSE" else 0.0 for r in results]
    labels = [classify(r) if r != "NO_POSE" else "no_pose" for r in results]

    # aggregate: average score + dominant label of last 3 s
    now = time.time()
    recent = [(r, s, l) for f, r, s, l in
              zip(frames, results, scores, labels)
              if now - f["t"] <= 3.0]
    if recent:
        avg = sum(s for _, s, _ in recent) / len(recent)
        dominant = max(set(l for _, _, l in recent),
                       key=lambda x: sum(1 for _, _, l2 in recent if l2 == x))
    else:
        avg = sum(scores) / len(scores)
        dominant = max(set(labels), key=labels.count)

    return jsonify({
        "device_id": device_id,
        "slot": slot_of_device.get(device_id),
        "score": round(avg, 1),           # 0-100
        "label": dominant,                # technique classification
        "frames_analyzed": len(results),
        "session_seconds": round(now - s.started_at, 1),
        "avg_metrics": {
            "torso_angle": round(sum(r["torso_angle"] for r in usable) / len(usable), 1),
            "hip_dev": round(sum(r["hip_dev"] for r in usable) / len(usable), 3),
        },
        "frames_total": len(frames),
        "usable_frames": len(usable),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)