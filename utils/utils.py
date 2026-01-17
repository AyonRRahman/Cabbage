import numpy as np
import cv2
import math
from collections import defaultdict
import os

def global_position(detections: list[tuple[int, int, int, int, float]], 
                    frame_meta: tuple[float, float, float], 
                    frame_idx: int, 
                    cabbage_positions: defaultdict[tuple[float, float], list[tuple[float, int]]], 
                    width: int, height: int, 
                    SENSOR_WIDTH_MM: float, FOCAL_LENGTH_MM: float, 
                    METERS_PER_DEG_LAT: float) -> defaultdict[tuple[float, float], list[tuple[float, int]]]:
    '''
    Converts pixel coordinates to global GPS coordinates.
    args:
        detections: List of detection tuples (x1, y1, x2, y2, confidence).
        frame_meta: Tuple containing (drone_lat, drone_lon, altitude).
        frame_idx: Index of the frame being processed.
        cabbage_positions: Dictionary to store cabbage positions.
        width: Width of the frame.
        height: Height of the frame.
        SENSOR_WIDTH_MM: Sensor width in mm.
        FOCAL_LENGTH_MM: Focal length in mm.
        METERS_PER_DEG_LAT: Meters per degree of latitude.
    '''
    drone_lat, drone_lon, alt = frame_meta
    gsd = (alt * SENSOR_WIDTH_MM) / (FOCAL_LENGTH_MM * width)

    for x1, y1, x2, y2, conf in detections:
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
        
    return cabbage_positions


def parse_srt(srt_file: str) -> dict:
    '''
    Parse the SRT file and extract frame data.
    Args:
        srt_file (str): The file path to the SRT file.
    Returns:
        dict: A dictionary containing frame data.
    '''
    
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


def annotate_frame(frame:np.ndarray, detections: list[tuple[int, int, int, int, float]],
                BOX_COLOR:tuple[int, int, int]=(0, 0, 255), 
                BOX_THICKNESS:int=2)->np.ndarray:
    '''
    Puts the given bboxes and their confidence scores on the frame.
    '''
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)

    return frame


def check_paths(ENGINE_PATH:str, VIDEO_PATH:str, SRT_PATH:str)->None:
    '''
    checks if the given paths exists
    '''

    for path in [ENGINE_PATH, VIDEO_PATH, SRT_PATH]:
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            exit(1)
        else:
            print(f"Path exists: {path}")


def create_output_writer(OUTPUT_VIDEO_PATH:str, fourcc:int, fps:float, width:int, height:int):
    '''
    Creates a video writer for the output video.
    '''
    return cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))


def open_video(VIDEO_PATH:str, OUTPUT_VIDEO_PATH:str, SAVE_OUT_VIDEO:bool=True):
    '''
    Opens the video and prepares the output video writer.
    '''
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Cannot open video"
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if SAVE_OUT_VIDEO:
        out = create_output_writer(OUTPUT_VIDEO_PATH, fourcc, fps, width, height)
    else:
        out = None

    return cap, out, width, height