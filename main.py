# ===================== Imports =====================

from collections import defaultdict


# ===================== Project related imports =====================
from utils.utils import check_paths, parse_srt, open_video, global_position, annotate_frame
from utils.inference import inference_single_frame, filter_positions, load_model
from utils.mapping import create_map, create_cluster_map
from config import Config

# ===================== CONFIG =====================
cfg = Config()

def main():
    check_paths(cfg.ENGINE_PATH, cfg.VIDEO_PATH, cfg.SRT_PATH)

    model = load_model(cfg.ENGINE_PATH)

    if model is None:
        print("Failed to load model.")
        return

    frame_meta = parse_srt(cfg.SRT_PATH)


    # Open video and prepare output writer
    cap, out, width, height = open_video(cfg.VIDEO_PATH, cfg.output_video_path(), cfg.SAVE_OUT_VIDEO)

    cabbage_positions = defaultdict(list)
    print("Processing video and projecting cabbages to map...")
    frame_idx = 0
    frame_used = 0

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx < cfg.START_MAPPING_FRAME) or (frame_idx % cfg.MAP_EVERY != 0):
            continue

        frame_used += 1
        if cfg.MAX_FRAMES_TO_MAP and frame_used == cfg.MAX_FRAMES_TO_MAP:
            break

        print(f"Processing frame {frame_idx}", end="\r")
        meta = frame_meta.get(frame_idx)
        if meta is None:
            print(f"No metadata for frame {frame_idx}, skipping...")
            continue

        detections = inference_single_frame(model, frame, cfg.DEVICE, cfg.CONF_THRES, cfg.DEDUPLICATE_DIST_THRES, cfg.SLICE_SIZE, cfg.OVERLAP_RATIO)
        cabbage_positions = global_position(detections, meta, frame_idx, cabbage_positions, width, height, cfg.SENSOR_WIDTH_MM, cfg.FOCAL_LENGTH_MM, cfg.METERS_PER_DEG_LAT)

        if cfg.SAVE_OUT_VIDEO:
            frame = annotate_frame(frame, detections, cfg.BOX_COLOR, cfg.BOX_THICKNESS)
            out.write(frame)


    cap.release()
    if cfg.SAVE_OUT_VIDEO:
        out.release()
    
    print("\nVideo processing complete.")
    print(f"Total cabbage positions found: {len(cabbage_positions)}")
    filtered_cabbage_positions = filter_positions(cabbage_positions, cfg.GRID_SIZE_METERS, cfg.METERS_PER_DEG_LAT)
    print(f"Filtered cabbage positions: {len(filtered_cabbage_positions)}")

    if cfg.SAVE_HTML_MAP:
        print("Creating HTML map...")
        if cfg.MAP_TYPE == 'cluster':
            create_cluster_map(filtered_cabbage_positions, cfg.map_output_path())
        elif cfg.MAP_TYPE == 'individual':
            create_map(filtered_cabbage_positions, cfg.map_output_path())

if __name__ == "__main__":
    main()