from dataclasses import dataclass

@dataclass
class Config:
    # ---------- IO ----------
    SAVE_OUT_VIDEO: bool = True
    SAVE_HTML_MAP: bool = True

    ENGINE_PATH: str = "best.engine"
    VIDEO_PATH: str = "/media/ayon/Windows/Users/User/Downloads/DJI_0797/DJI_0792.MP4"
    SRT_PATH: str = "/media/ayon/Windows/Users/User/Downloads/DJI_0797/DJI_0792.SRT"

    # ---------- Mapping ----------
    START_MAPPING_FRAME: int = 1
    MAP_EVERY: int = 50
    MAX_FRAMES_TO_MAP: int | None = None
    MAP_TYPE: str = "individual" # 'cluster' or 'individual'

    GRID_SIZE_METERS: float = 0.5
    METERS_PER_DEG_LAT: float = 111320

    # ---------- Detection ----------
    CONF_THRES: float = 0.25
    DEDUPLICATE_DIST_THRES: int = 25
    SLICE_SIZE: int = 640
    OVERLAP_RATIO: float = 0.2
    DEVICE: int = 0

    # ---------- Visualization ----------
    BOX_COLOR: tuple = (0, 0, 255)
    BOX_THICKNESS: int = 2

    # ---------- Camera ----------
    FOCAL_LENGTH_MM: float = 6.67
    SENSOR_WIDTH_MM: float = 10.26

    # ---------- Naming ----------
    def experiment_name(self):
        return f"{self.MAP_TYPE}_{self.MAP_EVERY}_DDT_{self.DEDUPLICATE_DIST_THRES}"

    def output_video_path(self):
        return f"output_geo_detected_{self.experiment_name()}.mp4"

    def map_output_path(self):
        return f"cabbage_field_map_{self.experiment_name()}.html"
