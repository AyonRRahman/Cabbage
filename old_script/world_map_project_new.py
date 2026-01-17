from pathlib import Path
import os
import cv2
import numbers
import math
try:
    import folium
    from folium.plugins import MarkerCluster
except ImportError:
    print("Folium not installed. Please install it via 'pip install folium' to enable map features.")

from ultralytics import YOLO
import numpy as np
from collections import defaultdict

import time

# Timing statistics
TIMINGS = defaultdict(list)


# ===================== CONFIG =====================
START_MAPPING_FRAME = 1 # start mapping from this frame number
MAP_EVERY = 1  # process every Nth frame for mapping
MAX_FRAMES_TO_MAP = None  # limit total frames processed for mapping (set None for no limit)
MAP_TYPE = 'individual'  # 'cluster' or 'individual'

# ────────────── Filtering parameters ──────────────
GRID_SIZE_METERS = 0.5  # Minimum distance between points (~50cm)
METERS_PER_DEG_LAT = 111320
# ==================================================
parent_folder = Path('/media/ayon/New Volume')
MODEL_PATH = parent_folder/"Cabbage_yolo11_drone_footage_no_process/runs/train/yolo11_cabbage_yolo11m_img640_b8_lr0.0005_freeze100/weights/best.pt"
VIDEO_PATH = "/media/ayon/Windows/Users/User/Downloads/DJI_0797/DJI_0792.MP4"
SRT_PATH = "/media/ayon/Windows/Users/User/Downloads/DJI_0797/DJI_0792.SRT" # or set manually if needed
OUTPUT_VIDEO_PATH = r"output_geo_detected.mp4"
MAP_OUTPUT_PATH = f"cabbage_field_map_{MAP_TYPE}_{MAP_EVERY}.html"
BENCHMARK_OUTPUT_PATH = f"benchmark_summary_{MAP_TYPE}_{MAP_EVERY}.txt"

#check paths
for path in [MODEL_PATH, VIDEO_PATH, SRT_PATH, OUTPUT_VIDEO_PATH, MAP_OUTPUT_PATH]:
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
    else:
        print(f"Path exists: {path}")

# ======================================================

CONF_THRES = 0.25
DEVICE = 0  # GPU

SLICE_SIZE = 640
OVERLAP_RATIO = 0.2

BOX_COLOR = (0, 0, 255)   # RED
BOX_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1

# DJI drone camera parameters (adjust if your model is different)
FOCAL_LENGTH_MM = 6.67      # 24mm equivalent cropped
SENSOR_WIDTH_MM = 10.26     # Approximate for common DJI sensors
# ==================================================

def timed(name):
    """Decorator to time functions"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            TIMINGS[name].append(dt)
            return result
        return wrapper
    return decorator

# Create interactive map with tiny individual markers
@timed("create_cluster_map")
def create_cluster_map(cabbage_positions, MAP_OUTPUT_PATH):
    if cabbage_positions:
        lats = [k[0] for k in cabbage_positions]
        lons = [k[1] for k in cabbage_positions]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=21,                  # Start very close
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            max_zoom=23                     # Allow maximum Google zoom
        )

        # Prepare data for FastMarkerCluster: list of [lat, lon, popup_html]
        marker_data = []
        for i,((cab_lat, cab_lon), records) in enumerate(cabbage_positions.items()):
            print(f"Adding marker {i+1}/{len(cabbage_positions)}", end="\r")
            max_conf = max(conf for conf, _ in records)
            frames_seen = sorted(set(frame for _, frame in records))
            popup_text = (
                f"<b>Cabbage</b><br>"
                f"Max Conf: {max_conf:.2f}<br>"
                f"Seen in {len(frames_seen)} frames<br>"
                f"First: {frames_seen[0]}, Last: {frames_seen[-1]}"
            )
            marker_data.append([cab_lat, cab_lon, popup_text])

        # FastMarkerCluster callback for tiny individual markers
        callback = """(
            function(row) {
                var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
                    radius: 1.5,        // Very small marker
                    color: '#ff0000',   // Red border
                    weight: 1,          // Thin border
                    fillColor: '#ff3333', // Slightly lighter red fill
                    fillOpacity: 0.9
                });
                marker.bindPopup(row[2]);
                return marker;
            }
        )"""

        from folium.plugins import FastMarkerCluster
        FastMarkerCluster(data=marker_data, callback=callback).add_to(m)

        # Optional: minify the HTML to further reduce file size
        m.save(MAP_OUTPUT_PATH)

        # Post-processing minification (optional but recommended)
        import re
        with open(MAP_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Shorten long UUIDs (Folium generates very long ones)
        html_content = re.sub(r'([0-9a-f]{8})[0-9a-f-]{12,}', r'\1', html_content)

        # Remove excess whitespace while keeping it readable
        lines = [line.rstrip() for line in html_content.split('\n')]
        html_content = '\n'.join(lines)

        with open(MAP_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Map saved: {MAP_OUTPUT_PATH}")
        print(f"Total unique cabbages mapped: {len(cabbage_positions)}")
        
    else:
        print("No cabbages detected.")

@timed("create_map")
def create_map(cabbage_positions, MAP_OUTPUT_PATH):
    if cabbage_positions:
        lats = [k[0] for k in cabbage_positions]
        lons = [k[1] for k in cabbage_positions]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=21,                  # Start very close
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            max_zoom=23                     # Allow maximum Google zoom
        )

        for i,((cab_lat, cab_lon), records) in enumerate(cabbage_positions.items()):
            print(f"Adding marker {i+1}/{len(cabbage_positions)}", end="\r")
            max_conf = max(conf for conf, _ in records)
            frames_seen = sorted(set(frame for _, frame in records))
            popup_text = f"<b>Cabbage</b><br>Max Conf: {max_conf:.2f}<br>Seen in {len(frames_seen)} frames<br>First: {frames_seen[0]}, Last: {frames_seen[-1]}"

            folium.CircleMarker(
                location=[cab_lat, cab_lon],
                radius=1.5,                     # Very small dot — adjust to 1 or 2 if needed
                color="#ff0000",                # Red border
                weight=1,                       # Thin border
                fill=True,
                fill_color="#ff3333",           # Slightly lighter red fill
                fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

        m.save(MAP_OUTPUT_PATH)
        print(f"Map saved: {MAP_OUTPUT_PATH}")
        print(f"Total unique cabbages mapped: {len(cabbage_positions)}")
        

    else:
        print("No cabbages detected.")

# Robust SRT parser (no regex, handles your format well)
@timed("parse_srt")
def parse_srt(srt_file):
    frame_data = {}
    with open(srt_file, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("\n\n")
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        frame_no = None
        lat = lon = alt = None

        for line in lines:
            if "FrameCnt:" in line:
                try:
                    frame_no = int(line.split("FrameCnt:")[1].split(",")[0].strip())
                except:
                    continue
            if "[" in line and "]" in line:
                parts = line.split("[")
                for part in parts[1:]:  # skip first empty
                    part = part.strip(" ]")
                    if "latitude:" in part:
                        try:
                            lat = float(part.split(":")[1].strip())
                        except:
                            pass
                    elif "longitude:" in part:
                        try:
                            lon = float(part.split(":")[1].strip())
                        except:
                            pass
                    elif "rel_alt:" in part:
                        try:
                            alt_str = part.split("rel_alt:")[1].split("abs_alt:")[0].strip()
                            alt = float(alt_str)
                        except:
                            pass

        if frame_no and lat is not None and lon is not None and alt is not None:
            frame_data[frame_no] = (lat, lon, alt)

    return frame_data

#sahi like slicing functions
@timed("slice_frame")
def slice_frame(frame, slice_size=640, overlap=0.2):
    H, W, _ = frame.shape
    stride = int(slice_size * (1 - overlap))
    slices = []
    coords = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            x2 = min(x + slice_size, W)
            y2 = min(y + slice_size, H)
            slices.append(frame[y:y2, x:x2])
            coords.append((x, y, x2 - x, y2 - y))
    return slices, coords

#merge predictions from slices back to original frame coords
@timed("merge_predictions")
def merge_predictions(results, coords):
    merged_boxes = []
    for r, (x_off, y_off, _, _) in zip(results, coords):
        if r.boxes is None:
            continue
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 += x_off; x2 += x_off
            y1 += y_off; y2 += y_off
            merged_boxes.append((x1, y1, x2, y2, conf))
    return merged_boxes

@timed("filter_positions")
def filter_positions(cabbage_positions, GRID_SIZE_METERS=0.5, METERS_PER_DEG_LAT=111320):
    valid_positions = []
    for k in cabbage_positions.keys():
        if isinstance(k, tuple) and len(k) == 2 and all(isinstance(v, numbers.Number) for v in k):
            valid_positions.append(k)
        else:
            print("Skipping invalid key:", k, type(k))

    if not valid_positions:
        raise ValueError("No valid positions found!")

    # ────────────── 2️⃣ Parameters for spatial hashing ──────────────
    
    # Approximate meters per degree longitude at the center latitude
    center_lat = sum(lat for lat, _ in valid_positions) / len(valid_positions)
    METERS_PER_DEG_LON = METERS_PER_DEG_LAT * math.cos(math.radians(center_lat))

    # ────────────── 3️⃣ Spatial hash function ──────────────
    def grid_hash(lat, lon):
        gx = int(lon * METERS_PER_DEG_LON / GRID_SIZE_METERS)
        gy = int(lat * METERS_PER_DEG_LAT / GRID_SIZE_METERS)
        return gx, gy

    # ────────────── 4️⃣ Filter points using grid ──────────────
    occupied_cells = set()
    filtered_cabbage_positions = defaultdict(list)

    for lat, lon in valid_positions:
        cell = grid_hash(lat, lon)
        if cell not in occupied_cells:
            filtered_cabbage_positions[(lat, lon)] = cabbage_positions[(lat, lon)]
            occupied_cells.add(cell)

    # ────────────── 5️⃣ Stats ──────────────
    print(f"Original unique positions: {len(valid_positions)}")
    print(f"Filtered positions: {len(filtered_cabbage_positions)}")

    return filtered_cabbage_positions

def print_benchmark_summary(TIMINGS, BENCHMARK_OUTPUT_PATH):
    summary_lines = []

    header = "\n================ BENCHMARK SUMMARY ================\n"
    print(header)
    summary_lines.append(header)

    for name, values in TIMINGS.items():
        if not values:
            continue

        total = sum(values)
        avg = total / len(values)
        mx = max(values)
        mn = min(values)

        if not name in ['total_run_time', 'total_cabbage_detected', 'total_cabbage_mapped']:
            line = (
                f"{name:25s} | "
                f"calls: {len(values):5d} | "
                f"avg: {avg*1000:8.2f} ms | "
                f"min: {mn*1000:8.2f} ms | "
            f"max: {mx*1000:8.2f} ms | "
            f"total: {total:8.2f} s"
            )
        else:
            line = f"{name:25s} | value: {total}"

        print(line)
        summary_lines.append(line + "\n")

    footer = "\n===================================================\n"
    print(footer)
    summary_lines.append(footer)

    # 🔹 Write to file
    with open(BENCHMARK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    print(f"Benchmark summary written to: {BENCHMARK_OUTPUT_PATH}")


    # for name, values in TIMINGS.items():
    #     if not values:
    #         continue

    #     total = sum(values)
    #     avg = total / len(values)
    #     mx = max(values)
    #     mn = min(values)

    #     line = (
    #         f"{name:25s} | "
    #         f"calls: {len(values):5d} | "
    #         f"avg: {avg*1000:8.2f} ms | "
    #         f"min: {mn*1000:8.2f} ms | "
    #         f"max: {mx*1000:8.2f} ms | "
    #         f"total: {total:8.2f} s"
    #     )

    #     print(line)
    #     summary_lines.append(line + "\n")

    # footer = "\n===================================================\n"
    # print(footer)
    # summary_lines.append(footer)

    # # 🔹 Write to file
    # with open(BENCHMARK_OUTPUT_PATH, "w", encoding="utf-8") as f:
    #     f.writelines(summary_lines)

    # print(f"Benchmark summary written to: {BENCHMARK_OUTPUT_PATH}")

if __name__ == "__main__":
    total_t0 = time.perf_counter()
    # Load model
    model = YOLO(MODEL_PATH)

    # Parse SRT
    print("Parsing SRT file...")
    frame_meta = parse_srt(SRT_PATH)
    print(f"Successfully parsed metadata for {len(frame_meta)} frames")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Cannot open video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # For map: store unique cabbages (rounded to ~10cm precision)
    cabbage_positions = defaultdict(list)  # key: (lat_round, lon_round) → list of (conf, frame)

    METERS_PER_DEG_LAT = 111320  # meters per degree latitude

    print("Processing video and projecting cabbages to map...")


    frame_idx = 0
    frame_used = 0
    while True:
        frame_t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
    # ===========Filtering Logic based on frame number =====================
        frame_idx += 1
        if frame_idx<START_MAPPING_FRAME:
            continue

        if not frame_idx % MAP_EVERY == 0:
            continue  # process every MAP_EVERY frame for speed
        
        frame_used += 1
        if MAX_FRAMES_TO_MAP and frame_used==MAX_FRAMES_TO_MAP:
            break

    # ======================================================================
        print(f"Processing frame {frame_idx}", end="\r")

        # Get metadata for this frame
        meta = frame_meta.get(frame_idx)
        if meta is None:
            # Optional: still draw detections, but no geo projection
            pass
        else:
            drone_lat, drone_lon, alt = meta

            # Ground Sampling Distance (meters per pixel)
            gsd = (alt * SENSOR_WIDTH_MM) / (FOCAL_LENGTH_MM * width)

            # Sliced inference
            slices, coords = slice_frame(frame, SLICE_SIZE, OVERLAP_RATIO)
            t0 = time.perf_counter()
            results = model.predict(slices, device=DEVICE, conf=CONF_THRES, verbose=False)
            TIMINGS["yolo_inference"].append(time.perf_counter() - t0)
            detections = merge_predictions(results, coords)

            t0 = time.perf_counter()
            for x1, y1, x2, y2, conf in detections:
                # Draw on video
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, BOX_COLOR, TEXT_THICKNESS)

                # Project center to ground
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                dx_px = cx - width / 2
                dy_px = cy - height / 2

                dx_m = dx_px * gsd
                dy_m = -dy_px * gsd  # image y-down = south

                dlat = dy_m / METERS_PER_DEG_LAT
                dlon = dx_m / (METERS_PER_DEG_LAT * math.cos(math.radians(drone_lat)))

                cab_lat = drone_lat + dlat
                cab_lon = drone_lon + dlon

                # Deduplicate: round to ~10cm precision
                key = (round(cab_lat, 7), round(cab_lon, 7))
                cabbage_positions[key].append((conf, frame_idx))

            TIMINGS["geo_projection"].append(time.perf_counter() - t0)

        out.write(frame)
        TIMINGS["per_frame_total"].append(time.perf_counter() - frame_t0)


    cap.release()
    out.release()
    print("\nVideo processing complete.")

    # Create interactive map
    filtered_cabbage_positions = filter_positions(cabbage_positions, GRID_SIZE_METERS, METERS_PER_DEG_LAT)

    if MAP_TYPE == 'cluster':
        create_cluster_map(filtered_cabbage_positions, MAP_OUTPUT_PATH)
    elif MAP_TYPE == 'individual':
        create_map(filtered_cabbage_positions, MAP_OUTPUT_PATH)


    TIMINGS["total_run_time"].append(time.perf_counter() - total_t0)
    TIMINGS["total_cabbage_detected"].append(len(cabbage_positions))
    TIMINGS["total_cabbage_mapped"].append(len(filtered_cabbage_positions))

    print_benchmark_summary(TIMINGS, BENCHMARK_OUTPUT_PATH)