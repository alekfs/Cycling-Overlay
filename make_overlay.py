from fitparse import FitFile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import time

# ---------- CONFIG ----------
FIT_FILENAME = "ride.fit"
OUTPUT_VIDEO = "overlay_greenscreen.mp4"
WIDTH, HEIGHT = 1920, 1080
FPS = 30  # 30fps for faster rendering

# Background (for chroma key)
BG_COLOR = (0, 255, 0)

# Clean color scheme
TEXT_MAIN = (255, 255, 255)
TEXT_SUBTLE = (220, 220, 220)
UNIT_COLOR = (180, 180, 180)

# Vibrant gradients
POWER_COLOR_LOW = (100, 200, 255)   # Bright blue
POWER_COLOR_MID = (150, 100, 255)   # Purple
POWER_COLOR_HIGH = (255, 50, 150)   # Pink

SPEED_COLOR_LOW = (0, 255, 200)     # Cyan
SPEED_COLOR_HIGH = (100, 150, 255)  # Blue

HR_COLOR_LOW = (255, 200, 0)        # Yellow
HR_COLOR_HIGH = (255, 50, 50)       # Red

CADENCE_COLOR_LOW = (255, 100, 200)  # Pink
CADENCE_COLOR_HIGH = (200, 50, 255)  # Purple

EASING_STRENGTH = 0.2


# ---------- FONT HELPERS ----------
def get_font(size: int, bold=False):
    try:
        if bold:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1)
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        try:
            return ImageFont.truetype("Arial.ttf", size)
        except:
            return ImageFont.load_default()


def get_text_bbox(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def three_point_gradient(low, mid, high, t):
    if t < 0.5:
        return lerp_color(low, mid, t * 2)
    else:
        return lerp_color(mid, high, (t - 0.5) * 2)


def smooth_value(current, target, easing=EASING_STRENGTH):
    if current is None:
        return target
    if target is None:
        return current
    return current + (target - current) * easing


def draw_text_with_shadow(draw, pos, text, font, fill, shadow_offset=3):
    """Simple drop shadow"""
    x, y = pos
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=fill)


# ---------- FIT PARSING ----------
def semicircles_to_degrees(v):
    return v * 180.0 / 2147483648.0


def load_fit_records(filename: str):
    fitfile = FitFile(filename)
    records = []

    for record in fitfile.get_messages("record"):
        fields = {f.name: f.value for f in record}
        ts = fields.get("timestamp")
        if ts is None:
            continue

        lat_raw = fields.get("position_lat")
        lon_raw = fields.get("position_long")
        lat_deg = semicircles_to_degrees(lat_raw) if lat_raw is not None else None
        lon_deg = semicircles_to_degrees(lon_raw) if lon_raw is not None else None

        records.append({
            "timestamp": ts,
            "power": fields.get("power"),
            "cadence": fields.get("cadence"),
            "speed": fields.get("speed"),
            "heart_rate": fields.get("heart_rate"),
            "lat": lat_deg,
            "lon": lon_deg,
            "distance": fields.get("distance"),
        })

    records.sort(key=lambda r: r["timestamp"])
    start_time = records[0]["timestamp"]
    for r in records:
        r["t"] = (r["timestamp"] - start_time).total_seconds()
    return records


def compute_stats(records):
    max_power = max_speed = max_cadence = max_hr = 0
    min_lat = max_lat = min_lon = max_lon = None
    last_dist = 0.0

    for r in records:
        if r["power"]: max_power = max(max_power, r["power"])
        if r["speed"]: max_speed = max(max_speed, r["speed"])
        if r["cadence"]: max_cadence = max(max_cadence, r["cadence"])
        if r["heart_rate"]: max_hr = max(max_hr, r["heart_rate"])
        if r["distance"]: last_dist = r["distance"]

        lat, lon = r["lat"], r["lon"]
        if lat and lon:
            min_lat = lat if min_lat is None else min(min_lat, lat)
            max_lat = lat if max_lat is None else max(max_lat, lat)
            min_lon = lon if min_lon is None else min(min_lon, lon)
            max_lon = lon if max_lon is None else max(max_lon, lon)

    return {
        "max_power": max(max_power, 600),
        "max_speed": max(max_speed, 20.0),
        "max_cadence": max(max_cadence, 120),
        "max_hr": max(max_hr, 180),
        "total_distance": last_dist,
        "min_lat": min_lat, "max_lat": max_lat,
        "min_lon": min_lon, "max_lon": max_lon,
    }


def interp_value(v0, v1, t0, t1, t):
    if v0 is None and v1 is None:
        return None
    if v0 is None:
        return v1
    if v1 is None:
        return v0
    if t1 <= t0:
        return v0
    f = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    return v0 + (v1 - v0) * f


def interpolate_record(records, t_target, last_index=0):
    n = len(records)
    if t_target <= records[0]["t"]:
        return records[0], 0
    if t_target >= records[-1]["t"]:
        return records[-1], n - 1

    i = last_index
    while i + 1 < n and records[i + 1]["t"] < t_target:
        i += 1

    r0, r1 = records[i], records[i + 1]
    t0, t1 = r0["t"], r1["t"]

    return {
        "t": t_target,
        "power": interp_value(r0["power"], r1["power"], t0, t1, t_target),
        "cadence": interp_value(r0["cadence"], r1["cadence"], t0, t1, t_target),
        "speed": interp_value(r0["speed"], r1["speed"], t0, t1, t_target),
        "heart_rate": interp_value(r0["heart_rate"], r1["heart_rate"], t0, t1, t_target),
        "distance": interp_value(r0["distance"], r1["distance"], t0, t1, t_target),
        "lat": interp_value(r0["lat"], r1["lat"], t0, t1, t_target),
        "lon": interp_value(r0["lon"], r1["lon"], t0, t1, t_target),
    }, i


def clamp01(x):
    if x is None:
        return 0.0
    return max(0.0, min(1.0, float(x)))


# ---------- GRADIENT CACHE ----------
def create_power_gradient(width, height):
    gradient = Image.new("RGB", (width, height))
    pixels = gradient.load()
    for x in range(width):
        t = x / max(1, width - 1)
        col = three_point_gradient(POWER_COLOR_LOW, POWER_COLOR_MID, POWER_COLOR_HIGH, t)
        for y in range(height):
            pixels[x, y] = col
    return gradient


# ---------- CLEAN DRAWING (NO BOXES) ----------
def draw_power_bar(img, draw, r, stats, fonts, smooth_state, gradient_cache):
    """Power bar - bottom center, no box"""
    power = r.get("power") or 0
    smooth_state["power"] = smooth_value(smooth_state.get("power", power), power, 0.2)
    display_power = smooth_state["power"]

    max_p = stats["max_power"]
    frac = clamp01(display_power / max_p)
    color_current = three_point_gradient(POWER_COLOR_LOW, POWER_COLOR_MID, POWER_COLOR_HIGH, frac)

    # Bar position - bottom center
    bar_width = 800
    bar_height = 22
    cx = WIDTH // 2
    bar_left = cx - bar_width // 2
    bar_top = HEIGHT - 120

    # Bar background
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_left + bar_width, bar_top + bar_height),
        radius=11,
        fill=(30, 30, 30)
    )

    # Gradient fill
    fill_width = int(bar_width * frac)
    if fill_width > 0:
        mask = Image.new("L", (bar_width, bar_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle((0, 0, fill_width, bar_height), radius=11, fill=255)
        img.paste(gradient_cache["power"], (bar_left, bar_top), mask)

    # Power value - ABOVE bar with shadow
    power_txt = str(int(round(display_power)))
    w, h = get_text_bbox(draw, power_txt, fonts["huge"])
    draw_text_with_shadow(draw, (cx - w / 2, bar_top - h - 15), power_txt, fonts["huge"], color_current, 4)

    # "W" unit next to number
    unit_x = cx + w / 2 + 8
    draw.text((unit_x, bar_top - h - 5), "W", font=fonts["medium"], fill=TEXT_SUBTLE)


def draw_speed_gauge(draw, cx, cy, r, stats, fonts, smooth_state):
    """Clean speed gauge - left side, simple rounded style like power bar"""
    speed_mps = r.get("speed") or 0
    speed_kph = speed_mps * 3.6
    smooth_state["speed"] = smooth_value(smooth_state.get("speed", speed_kph), speed_kph, 0.15)
    display_speed = smooth_state["speed"]

    max_speed_kph = stats["max_speed"] * 3.6
    frac = clamp01(display_speed / max_speed_kph)
    color = lerp_color(SPEED_COLOR_LOW, SPEED_COLOR_HIGH, frac)

    radius = 160
    start_angle = 135
    end_angle = 405

    # Background arc - PIL arcs have natural rounded ends
    draw.arc((cx - radius, cy - radius, cx + radius, cy + radius),
             start=start_angle, end=end_angle, fill=(80, 80, 80), width=16)

    # Active arc - PIL arcs have natural rounded ends
    if frac > 0:
        current_angle = start_angle + int(270 * frac)
        draw.arc((cx - radius, cy - radius, cx + radius, cy + radius),
                 start=start_angle, end=current_angle, fill=color, width=20)

    # Value with shadow
    val_txt = str(int(round(display_speed)))
    w, h = get_text_bbox(draw, val_txt, fonts["big"])
    draw_text_with_shadow(draw, (cx - w / 2, cy - h / 2), val_txt, fonts["big"], TEXT_MAIN, 3)

    # Unit below
    unit_w, _ = get_text_bbox(draw, "km/h", fonts["small"])
    draw.text((cx - unit_w / 2, cy + h / 2 + 10), "km/h", font=fonts["small"], fill=UNIT_COLOR)


def draw_cadence_gauge(draw, cx, cy, r, stats, fonts, smooth_state):
    """Clean cadence gauge - right side, simple rounded style like power bar"""
    cad = r.get("cadence") or 0
    smooth_state["cadence"] = smooth_value(smooth_state.get("cadence", cad), cad, 0.15)
    display_cad = smooth_state["cadence"]

    max_cad = stats["max_cadence"]
    frac = clamp01(display_cad / max_cad)
    color = lerp_color(CADENCE_COLOR_LOW, CADENCE_COLOR_HIGH, frac)

    radius = 160
    start_angle = 135
    end_angle = 405

    # Background arc - PIL arcs have natural rounded ends
    draw.arc((cx - radius, cy - radius, cx + radius, cy + radius),
             start=start_angle, end=end_angle, fill=(80, 80, 80), width=16)

    # Active arc - PIL arcs have natural rounded ends
    if frac > 0:
        current_angle = start_angle + int(270 * frac)
        draw.arc((cx - radius, cy - radius, cx + radius, cy + radius),
                 start=start_angle, end=current_angle, fill=color, width=20)

    # Value with shadow
    val_txt = str(int(round(display_cad)))
    w, h = get_text_bbox(draw, val_txt, fonts["big"])
    draw_text_with_shadow(draw, (cx - w / 2, cy - h / 2), val_txt, fonts["big"], TEXT_MAIN, 3)

    # Unit below
    unit_w, _ = get_text_bbox(draw, "rpm", fonts["small"])
    draw.text((cx - unit_w / 2, cy + h / 2 + 10), "rpm", font=fonts["small"], fill=UNIT_COLOR)


def draw_heart_rate(draw, r, stats, fonts, smooth_state):
    """Heart rate - top center, no overlap"""
    hr = r.get("heart_rate")
    if hr is not None:
        smooth_state["hr"] = smooth_value(smooth_state.get("hr", hr), hr, 0.2)
        display_hr = smooth_state["hr"]
    else:
        display_hr = None

    max_hr = stats["max_hr"]
    frac = clamp01((display_hr or 0) / max_hr)
    color = lerp_color(HR_COLOR_LOW, HR_COLOR_HIGH, frac)

    # Position - top center (well above power)
    cx = WIDTH // 2
    y = 80  # Top of screen

    if display_hr:
        hr_txt = str(int(round(display_hr)))
        w, h = get_text_bbox(draw, hr_txt, fonts["medium"])

        # Just the heart rate number with shadow
        draw_text_with_shadow(draw, (cx - w / 2, y), hr_txt, fonts["medium"], color, 3)

        # BPM unit below
        unit_w, _ = get_text_bbox(draw, "bpm", fonts["tiny"])
        draw.text((cx - unit_w / 2, y + h + 5), "bpm", font=fonts["tiny"], fill=UNIT_COLOR)


def draw_gps_map(draw, r, all_points, trail_cache, stats, smooth_state):
    """Clean GPS map - top left, no background"""
    if not all_points:
        return

    # Position - top left corner
    pad = 30
    size = 300
    x0, y0 = pad, pad

    # No background - just draw the route directly on green screen

    # Full route in gray - always draw
    if len(all_points) > 1:
        draw.line(all_points, fill=(70, 70, 70), width=3)

    # Completed trail in cyan - find closest timestamp
    current_t = r.get("t", 0)

    # Find the closest cached trail (interpolate between cached points)
    trail = None
    for cached_t, cached_trail in sorted(trail_cache.items()):
        if cached_t <= current_t:
            trail = cached_trail
        else:
            break

    # Always draw trail if we have any
    if trail and len(trail) > 1:
        draw.line(trail, fill=(0, 255, 255), width=5)

    # Current position marker - keep last known position if GPS data missing
    if trail and len(trail) > 0:
        px, py = trail[-1]

        # Initialize or update smoothed position
        if "gps_x" not in smooth_state or "gps_y" not in smooth_state:
            smooth_state["gps_x"], smooth_state["gps_y"] = px, py
        else:
            # Smooth the position
            smooth_state["gps_x"] = smooth_value(smooth_state["gps_x"], px, 0.3)
            smooth_state["gps_y"] = smooth_value(smooth_state["gps_y"], py, 0.3)

        cx, cy = smooth_state["gps_x"], smooth_state["gps_y"]

        # Always draw marker (no flashing)
        radius = 7
        draw.ellipse((cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1),
                    fill=(255, 255, 255))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=(255, 215, 0))
    elif "gps_x" in smooth_state and "gps_y" in smooth_state:
        # Keep drawing at last known position if GPS data is missing
        cx, cy = smooth_state["gps_x"], smooth_state["gps_y"]
        radius = 7
        draw.ellipse((cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1),
                    fill=(255, 255, 255))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=(255, 215, 0))


def precompute_gps_data(records, stats):
    min_lat, max_lat = stats["min_lat"], stats["max_lat"]
    min_lon, max_lon = stats["min_lon"], stats["max_lon"]

    if None in (min_lat, max_lat, min_lon, max_lon):
        return [], {}

    pad, size = 30, 300
    x0, y0 = pad, pad

    lat_span = max_lat - min_lat or 1.0
    lon_span = max_lon - min_lon or 1.0

    def project(lat, lon):
        u = (lon - min_lon) / lon_span
        v = (lat - min_lat) / lat_span
        return (x0 + u * size, y0 + size - v * size)

    all_points = []
    trail_cache = {}
    current_trail = []

    for rec in records:
        lat, lon = rec["lat"], rec["lon"]
        if lat and lon:
            pt = project(lat, lon)
            all_points.append(pt)
            current_trail.append(pt)
            trail_cache[rec["t"]] = list(current_trail)

    return all_points, trail_cache


# ---------- MAIN ----------
def main():
    print(f"Loading FIT file: {FIT_FILENAME}")
    start_time = time.time()

    records = load_fit_records(FIT_FILENAME)
    stats = compute_stats(records)

    total_time = records[-1]["t"]
    total_frames = int(total_time * FPS) + 1

    print(f"Ride: {total_time / 60:.1f} min | Frames: {total_frames:,} @ {FPS} fps")
    print(f"Pre-processing...")

    all_gps, trail_cache = precompute_gps_data(records, stats)
    gradient_cache = {"power": create_power_gradient(800, 22)}

    fonts = {
        "huge": get_font(70, bold=True),
        "big": get_font(56, bold=True),
        "medium": get_font(38, bold=True),
        "small": get_font(24),
        "tiny": get_font(18),
    }

    print(f"Initializing video writer...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")

    smooth_state = {}
    last_index = 0
    last_update = time.time()

    print(f"\nRendering overlay...\n")

    for frame in range(total_frames):
        t = frame / FPS
        r, last_index = interpolate_record(records, t, last_index)

        # Create frame
        img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Draw all elements (no boxes!)
        draw_gps_map(draw, r, all_gps, trail_cache, stats, smooth_state)
        draw_heart_rate(draw, r, stats, fonts, smooth_state)  # Top center
        draw_power_bar(img, draw, r, stats, fonts, smooth_state, gradient_cache)  # Bottom center
        draw_speed_gauge(draw, 240, HEIGHT - 220, r, stats, fonts, smooth_state)  # Bottom left
        draw_cadence_gauge(draw, WIDTH - 240, HEIGHT - 220, r, stats, fonts, smooth_state)  # Bottom right

        # Write frame
        frame_np = np.array(img)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        # Progress
        now = time.time()
        if now - last_update >= 2.0:
            elapsed = now - start_time
            pct = (frame / total_frames) * 100
            fps_actual = frame / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame) / fps_actual if fps_actual > 0 else 0
            print(f"  {pct:5.1f}% | Frame {frame:6,}/{total_frames:,} | {fps_actual:.1f} fps | ETA: {eta/60:.1f} min")
            last_update = now

    writer.release()
    total = time.time() - start_time
    print(f"\n✓ Overlay complete in {total / 60:.1f} minutes!")
    print(f"  Output: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
