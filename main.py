import argparse
from src.pipeline import run_detector_tracker_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + StrongSORT Tracker")
    parser.add_argument('--input', required=True, help="Path to input video")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_detector_tracker_pipeline(args.input) 