import torch
import numpy as np
from scipy.spatial import cKDTree
import numbers
import math
from collections import defaultdict
from ultralytics import YOLO

def load_model(ENGINE_PATH:str):
    print(f"Loading TensorRT engine: {ENGINE_PATH}")
    model = YOLO(ENGINE_PATH)
    print("TensorRT engine loaded successfully")
    
    return model

def slice_frame(frame:np.ndarray, slice_size:int=640, overlap:float=0.2)->tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    '''
    Slices the input frame into smaller patches.
    Args:
        frame: The input image frame to slice.
        slice_size: The size of each square slice.
        overlap: The fraction of overlap between slices.
    Returns:
        A tuple containing:
            - A list of sliced image patches.
            - A list of coordinates for each patch.
    '''
    assert frame.ndim == 3, "Frame must be a HxWxC image"
    assert isinstance(slice_size, int) and slice_size > 0, "slice_size must be a positive integer"
    assert isinstance(overlap, float) and 0 <= overlap < 1, "overlap must be a float between 0 and 1"

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


def merge_predictions_vectorized(results: list, coords: list) -> list[tuple[int, int, int, int, float]]:
    """
    Merge YOLO results from tiles or batches into global coordinates.
    Returns a list of tuples: (x1, y1, x2, y2, conf) with ints/floats.
    args:
        results: A list of detection results from the model.
        coords: A list of coordinates corresponding to each result.
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


def merge_predictions(results: list, coords: list) -> list[tuple[int, int, int, int, float]]:
    """
    Merge YOLO results from tiles or batches into global coordinates.
    Returns a list of tuples: (x1, y1, x2, y2, conf) with ints/floats.
    args:
        results: A list of detection results from the model.
        coords: A list of coordinates corresponding to each result.
    """
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

def deduplicate_by_distance(detections: list[tuple[int, int, int, int, float]],
                            dist_thresh_px:int=25) -> list[tuple[int, int, int, int, float]]:
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

def deduplicate_by_grid(detections: list[tuple[int, int, int, int, float]],
                        cell_size_px:int=25) -> list[tuple[int, int, int, int, float]]:
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

def deduplicate_by_distance_kdtree(detections: list[tuple[int, int, int, int, float]],
                                dist_thresh_px:int=25) -> list[tuple[int, int, int, int, float]]:
    '''
    Removes duplicate detections of the same cabbage in the same frame based on distance using a KDTree. 
    like non-maximum suppression.
    args:
        detections: List of detection tuples (x1, y1, x2, y2, conf)
        dist_thresh_px: Distance threshold in pixels
    '''
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


def filter_positions(cabbage_positions:defaultdict[list],
                    GRID_SIZE_METERS:float=0.5,
                    METERS_PER_DEG_LAT:int=111320) -> defaultdict[list]:
    '''
    Filters cabbage positions based on grid hashing.
    '''
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

def inference_single_frame(model, frame, device, conf_thres, dist_thresh_px, slice_size, overlap_ratio):
    slices, coords = slice_frame(frame, slice_size, overlap_ratio)
    results = model.predict(slices, device=device, conf=conf_thres, verbose=False)
    detections = merge_predictions_vectorized(results, coords)
    detections = deduplicate_by_distance_kdtree(detections, dist_thresh_px=dist_thresh_px)
    return detections