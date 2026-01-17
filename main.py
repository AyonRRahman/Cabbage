# ===================== Imports =====================

import os
from collections import defaultdict


# ===================== Project related imports =====================
from utils.utils import check_paths, parse_srt, open_video, global_position, annotate_frame
from utils.inference import inference_single_frame, filter_positions, load_model
from utils.mapping import create_map, create_cluster_map


# ===================== CONFIG =====================
SAVE_OUT_VIDEO = False
SAVE_HTML_MAP = False
START_MAPPING_FRAME = 1
MAP_EVERY = 500
MAX_FRAMES_TO_MAP = None
MAP_TYPE = 'individual'

GRID_SIZE_METERS = 0.5
METERS_PER_DEG_LAT = 111320


ENGINE_PATH = "best.engine"  # ← Use the TensorRT engine we exported

VIDEO_PATH = "/media/ayon/Windows/Users/User/Downloads/DJI_0797/DJI_0792.MP4"
SRT_PATH = "/media/ayon/Windows/Users/User/Downloads/DJI_0797/DJI_0792.SRT"

DEDUPLICATE_DIST_THRES = 25

#change naming as needed
EXP_NAMING = f"{MAP_TYPE}_{MAP_EVERY}_trt_new_merge_DDT_{DEDUPLICATE_DIST_THRES}_KDTREE"
OUTPUT_VIDEO_PATH = f"output_geo_detected_trt_new_merge_{EXP_NAMING}.mp4"  
MAP_OUTPUT_PATH = f"cabbage_field_map_{EXP_NAMING}.html"
# BENCHMARK_OUTPUT_PATH = f"benchmark_summary_{EXP_NAMING}.txt"

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



def main():
    check_paths(ENGINE_PATH, VIDEO_PATH, SRT_PATH)
    
    model = load_model(ENGINE_PATH)

    if model is None:
        print("Failed to load model.")
        return
    
    frame_meta = parse_srt(SRT_PATH)


    # Open video and prepare output writer
    cap, out, width, height = open_video(VIDEO_PATH, OUTPUT_VIDEO_PATH, SAVE_OUT_VIDEO)

    cabbage_positions = defaultdict(list)
    print("Processing video and projecting cabbages to map...")
    frame_idx = 0
    frame_used = 0

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx < START_MAPPING_FRAME) or (frame_idx % MAP_EVERY != 0):
            continue

        frame_used += 1
        if MAX_FRAMES_TO_MAP and frame_used == MAX_FRAMES_TO_MAP:
            break

        print(f"Processing frame {frame_idx}", end="\r")
        meta = frame_meta.get(frame_idx)
        if meta is None:
            print(f"No metadata for frame {frame_idx}, skipping...")
            continue

        detections = inference_single_frame(model, frame, DEVICE, CONF_THRES, DEDUPLICATE_DIST_THRES, SLICE_SIZE, OVERLAP_RATIO)
        cabbage_positions = global_position(detections, meta, frame_idx, cabbage_positions, width, height, SENSOR_WIDTH_MM, FOCAL_LENGTH_MM, METERS_PER_DEG_LAT)

        if SAVE_OUT_VIDEO:
            frame = annotate_frame(frame, detections, BOX_COLOR, BOX_THICKNESS)
            out.write(frame)


    cap.release()
    if SAVE_OUT_VIDEO:
        out.release()
    
    print("\nVideo processing complete.")
    print(f"Total cabbage positions found: {len(cabbage_positions)}")
    filtered_cabbage_positions = filter_positions(cabbage_positions, GRID_SIZE_METERS, METERS_PER_DEG_LAT)
    print(f"Filtered cabbage positions: {len(filtered_cabbage_positions)}")

    if SAVE_HTML_MAP:
        print("Creating HTML map...")
        if MAP_TYPE == 'cluster':
            create_cluster_map(filtered_cabbage_positions, MAP_OUTPUT_PATH)
        elif MAP_TYPE == 'individual':
            create_map(filtered_cabbage_positions, MAP_OUTPUT_PATH)

if __name__ == "__main__":
    main()