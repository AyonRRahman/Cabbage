from pathlib import Path
import os
import cv2
import numbers
import math
import torch
import torchvision.ops as ops
from scipy.spatial import cKDTree

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
START_MAPPING_FRAME = 1
MAP_EVERY = 1
MAX_FRAMES_TO_MAP = None
MAP_TYPE = 'individual'

GRID_SIZE_METERS = 0.5
METERS_PER_DEG_LAT = 111320


ENGINE_PATH = Path("best.engine")  # ← Use the TensorRT engine we exported

VIDEO_PATH = "/media/ayon/Windows/Users/User/Downloads/DJI_0797/DJI_0792.MP4"
SRT_PATH = "/media/ayon/Windows/Users/User/Downloads/DJI_0797/DJI_0792.SRT"

DEDUPLICATE_DIST_THRES = 25
EXP_NAMING = f"{MAP_TYPE}_{MAP_EVERY}_trt_new_merge_DDT_{DEDUPLICATE_DIST_THRES}_KDTREE"

OUTPUT_VIDEO_PATH = f"output_geo_detected_trt_new_merge_{EXP_NAMING}.mp4"  
MAP_OUTPUT_PATH = f"cabbage_field_map_{EXP_NAMING}.html"
BENCHMARK_OUTPUT_PATH = f"benchmark_summary_{EXP_NAMING}.txt"

# Check paths
for path in [ENGINE_PATH, VIDEO_PATH, SRT_PATH]:
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        exit(1)
    else:
        print(f"Path exists: {path}")

CONF_THRES = 0.25
DEVICE = 0  # GPU
SLICE_SIZE = 640
OVERLAP_RATIO = 0.2
BOX_COLOR = (0, 0, 255)  # RED
BOX_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1

FOCAL_LENGTH_MM = 6.67
SENSOR_WIDTH_MM = 10.26

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

@timed("create_cluster_map")
def create_cluster_map(cabbage_positions, MAP_OUTPUT_PATH):
    if cabbage_positions:
        lats = [k[0] for k in cabbage_positions]
        lons = [k[1] for k in cabbage_positions]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=21,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            max_zoom=23
        )
        marker_data = []
        for i, ((cab_lat, cab_lon), records) in enumerate(cabbage_positions.items()):
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

        from folium.plugins import FastMarkerCluster
        callback = """(
            function(row) {
                var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
                    radius: 1.5,
                    color: '#ff0000',
                    weight: 1,
                    fillColor: '#ff3333',
                    fillOpacity: 0.9
                });
                marker.bindPopup(row[2]);
                return marker;
            }
        )"""
        FastMarkerCluster(data=marker_data, callback=callback).add_to(m)

        m.save(MAP_OUTPUT_PATH)
        # Minification (optional)
        import re
        with open(MAP_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            html = f.read()
        html = re.sub(r'([0-9a-f]{8})[0-9a-f-]{12,}', r'\1', html)
        lines = [line.rstrip() for line in html.split('\n')]
        with open(MAP_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

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
            zoom_start=21,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            max_zoom=23
        )
        for i, ((cab_lat, cab_lon), records) in enumerate(cabbage_positions.items()):
            print(f"Adding marker {i+1}/{len(cabbage_positions)}", end="\r")
            max_conf = max(conf for conf, _ in records)
            frames_seen = sorted(set(frame for _, frame in records))
            popup_text = f"<b>Cabbage</b><br>Max Conf: {max_conf:.2f}<br>Seen in {len(frames_seen)} frames<br>First: {frames_seen[0]}, Last: {frames_seen[-1]}"
            folium.CircleMarker(
                location=[cab_lat, cab_lon],
                radius=1.5,
                color="#ff0000",
                weight=1,
                fill=True,
                fill_color="#ff3333",
                fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

        m.save(MAP_OUTPUT_PATH)
        print(f"Map saved: {MAP_OUTPUT_PATH}")
        print(f"Total unique cabbages mapped: {len(cabbage_positions)}")
    else:
        print("No cabbages detected.")

@timed("parse_srt")
def parse_srt(srt_file):
    frame_data = {}
    with open(srt_file, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = content.split("\n\n")
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        frame_no = lat = lon = alt = None
        for line in lines:
            if "FrameCnt:" in line:
                try:
                    frame_no = int(line.split("FrameCnt:")[1].split(",")[0].strip())
                except:
                    continue
            if "[" in line and "]" in line:
                parts = line.split("[")
                for part in parts[1:]:
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

@timed("merge_predictions")
def merge_predictions_vectorized(results, coords):
    """
    Merge YOLO results from tiles or batches into global coordinates.
    Returns a list of tuples: (x1, y1, x2, y2, conf) with ints/floats.
    """
    merged_boxes = []

    for r, (x_off, y_off, _, _) in zip(results, coords):
        if r.boxes is None or len(r.boxes) == 0:
            continue

        # Stack all box info at once
        cls = torch.stack([b.cls for b in r.boxes]).squeeze(-1)       # shape: (N,)
        conf = torch.stack([b.conf for b in r.boxes]).squeeze(-1)     # shape: (N,)
        xyxy = torch.stack([b.xyxy for b in r.boxes]).squeeze(1)      # shape: (N,4)

        # Filter class 0
        mask = cls == 0
        if mask.sum() == 0:
            continue

        xyxy = xyxy[mask].cpu().numpy().astype(int)  # convert to CPU numpy ints
        conf = conf[mask].cpu().numpy().astype(float)

        # Add offsets
        xyxy[:, 0] += x_off  # x1
        xyxy[:, 2] += x_off  # x2
        xyxy[:, 1] += y_off  # y1
        xyxy[:, 3] += y_off  # y2

        # Combine xyxy and conf into merged_boxes format
        merged_boxes.extend([(*xy, c) for xy, c in zip(xyxy, conf)])

    return merged_boxes

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

def deduplicate_by_distance(detections, dist_thresh_px=25):
    if not detections:
        return []

    dets = np.array(detections)
    centers = np.column_stack((
        (dets[:, 0] + dets[:, 2]) / 2,
        (dets[:, 1] + dets[:, 3]) / 2
    ))
    scores = dets[:, 4]

    keep = []
    used = np.zeros(len(dets), dtype=bool)

    for i in np.argsort(-scores):
        if used[i]:
            continue

        keep.append(detections[i])

        dists = np.linalg.norm(centers - centers[i], axis=1)
        used |= dists < dist_thresh_px

    return keep

def deduplicate_by_grid(detections, cell_size_px=25):
    if not detections:
        return []

    grid = {}

    for x1, y1, x2, y2, conf in detections:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        gx = int(cx // cell_size_px)
        gy = int(cy // cell_size_px)

        key = (gx, gy)

        if key not in grid or conf > grid[key][4]:
            grid[key] = (x1, y1, x2, y2, conf)

    return list(grid.values())

def deduplicate_by_distance_kdtree(detections, dist_thresh_px=25):
    if not detections:
        return []

    dets = np.asarray(detections, dtype=np.float32)
    centers = np.column_stack((
        (dets[:, 0] + dets[:, 2]) * 0.5,
        (dets[:, 1] + dets[:, 3]) * 0.5
    ))
    scores = dets[:, 4]

    tree = cKDTree(centers)

    order = np.argsort(-scores)
    used = np.zeros(len(dets), dtype=bool)
    keep = []

    for i in order:
        if used[i]:
            continue

        keep.append(tuple(detections[i]))
        idxs = tree.query_ball_point(centers[i], dist_thresh_px)
        used[idxs] = True

    return keep

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

    center_lat = sum(lat for lat, _ in valid_positions) / len(valid_positions)
    METERS_PER_DEG_LON = METERS_PER_DEG_LAT * math.cos(math.radians(center_lat))

    def grid_hash(lat, lon):
        gx = int(lon * METERS_PER_DEG_LON / GRID_SIZE_METERS)
        gy = int(lat * METERS_PER_DEG_LAT / GRID_SIZE_METERS)
        return gx, gy

    occupied_cells = set()
    filtered_cabbage_positions = defaultdict(list)
    for lat, lon in valid_positions:
        cell = grid_hash(lat, lon)
        if cell not in occupied_cells:
            filtered_cabbage_positions[(lat, lon)] = cabbage_positions[(lat, lon)]
            occupied_cells.add(cell)

    print(f"Original unique positions: {len(valid_positions)}")
    print(f"Filtered positions: {len(filtered_cabbage_positions)}")
    return filtered_cabbage_positions

def print_benchmark_summary(TIMINGS, BENCHMARK_OUTPUT_PATH):
    summary_lines = []
    header = "\n================ BENCHMARK SUMMARY (TensorRT) ================\n"
    print(header)
    summary_lines.append(header)
    for name, values in TIMINGS.items():
        if not values:
            continue
        total = sum(values)
        avg = total / len(values)
        mx = max(values)
        mn = min(values)
        if name not in ['total_run_time', 'total_cabbage_detected', 'total_cabbage_mapped']:
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

    with open(BENCHMARK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print(f"Benchmark summary written to: {BENCHMARK_OUTPUT_PATH}")




if __name__ == "__main__":
    total_t0 = time.perf_counter()

    # ── Load TensorRT engine instead of .pt model ───────────────────────
    if not ENGINE_PATH.is_file():
        print(f"Error: TensorRT engine not found: {ENGINE_PATH}")
        exit(1)

    print(f"Loading TensorRT engine: {ENGINE_PATH}")
    engine_load_t0 = time.perf_counter()
    model = YOLO(str(ENGINE_PATH))
    TIMINGS["engine_load"].append(time.perf_counter() - engine_load_t0)
    print("TensorRT engine loaded successfully")

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

    cabbage_positions = defaultdict(list)
    print("Processing video and projecting cabbages to map...")
    frame_idx = 0
    frame_used = 0

    while True:
        frame_t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx < START_MAPPING_FRAME:
            continue
        if frame_idx % MAP_EVERY != 0:
            continue

        frame_used += 1
        if MAX_FRAMES_TO_MAP and frame_used == MAX_FRAMES_TO_MAP:
            break

        print(f"Processing frame {frame_idx}", end="\r")

        meta = frame_meta.get(frame_idx)
        if meta is not None:
            drone_lat, drone_lon, alt = meta
            gsd = (alt * SENSOR_WIDTH_MM) / (FOCAL_LENGTH_MM * width)

            slices, coords = slice_frame(frame, SLICE_SIZE, OVERLAP_RATIO)
            t0 = time.perf_counter()
            results = model.predict(slices, device=DEVICE, conf=CONF_THRES, verbose=False)
            TIMINGS["yolo_inference"].append(time.perf_counter() - t0)

            detections = merge_predictions_vectorized(results, coords)
            #optional deduplication for making the export video better. Adds a bit of time.
            # detections = deduplicate_by_distance(detections, dist_thresh_px=DEDUPLICATE_DIST_THRES)
            # detections = deduplicate_by_grid(detections, cell_size_px=DEDUPLICATE_DIST_THRES)
            detections = deduplicate_by_distance_kdtree(detections, dist_thresh_px=DEDUPLICATE_DIST_THRES)

            t0 = time.perf_counter()

            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, BOX_COLOR, TEXT_THICKNESS)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dx_px = cx - width / 2
                dy_px = cy - height / 2
                dx_m = dx_px * gsd
                dy_m = -dy_px * gsd
                dlat = dy_m / METERS_PER_DEG_LAT
                dlon = dx_m / (METERS_PER_DEG_LAT * math.cos(math.radians(drone_lat)))
                cab_lat = drone_lat + dlat
                cab_lon = drone_lon + dlon

                key = (round(cab_lat, 7), round(cab_lon, 7))
                cabbage_positions[key].append((conf, frame_idx))

            TIMINGS["geo_projection"].append(time.perf_counter() - t0)

        out.write(frame)
        TIMINGS["per_frame_total"].append(time.perf_counter() - frame_t0)

    cap.release()
    out.release()
    print("\nVideo processing complete.")

    # Create map with filtered positions
    filtered_cabbage_positions = filter_positions(cabbage_positions, GRID_SIZE_METERS, METERS_PER_DEG_LAT)

    if MAP_TYPE == 'cluster':
        create_cluster_map(filtered_cabbage_positions, MAP_OUTPUT_PATH)
    elif MAP_TYPE == 'individual':
        create_map(filtered_cabbage_positions, MAP_OUTPUT_PATH)

    TIMINGS["total_run_time"].append(time.perf_counter() - total_t0)
    TIMINGS["total_cabbage_detected"].append(len(cabbage_positions))
    TIMINGS["total_cabbage_mapped"].append(len(filtered_cabbage_positions))

    print_benchmark_summary(TIMINGS, BENCHMARK_OUTPUT_PATH)



