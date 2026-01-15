# Cabbage Project 🥬

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/conda-environment-green)](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

## Overview

This project detects cabbages in video frames and generates maps and annotated outputs.  
It supports both **TensorRT FP16 inference** and **PyTorch saved models**.

---

## Setup and Usage

Activate your environment and run everything with these commands:

```bash
# 1. Create and activate the Conda environment
conda env create -f environment.yml
conda activate <your-environment-name>

# 2. Create the TensorRT FP16 engine
bash create_TensorRT.sh

# 3. Run the TensorRT inference script to generate outputs
python inference_map_tensorrt.py

# 4. Torch inference (optional, slightly outdated)
python world_map_project_new.py

---

## Output
Running the scripts will generate the following files:
1 .mp4 — Video with cabbage annotations

2 .html — Map of detected cabbages

3 .txt — Timing and related information
