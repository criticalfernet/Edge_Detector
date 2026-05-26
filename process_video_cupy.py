import cv2
import time
import argparse
import numpy as np
from multi_scale_cupy import multi_scale_LOG_cupy_frame

def process_video(input_path, output_path, sigmas=[0.85, 1.2, 1.8, 2.8, 4.5], base_t=0.6):
    print(f"Opening video file: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        fps = 24.0

    print(f"Video Info -> {width}x{height} @ {fps}fps | Total Frames: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    frame_count = 0
    start_time = time.time()
    
    print("Warming up Python CuPy API...")
    _ = multi_scale_LOG_cupy_frame(np.zeros((height, width), dtype=np.uint8), sigmas, base_t)

    print(f"Beginning Hardware Processing...\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge_map = multi_scale_LOG_cupy_frame(gray_frame, sigmas, base_t)

            out.write(edge_map)
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames... (Avg Speed: {avg_fps:.2f} fps)")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user!")

    finally:
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        print(f"\nDone! Processed {frame_count} frames in {total_time:.2f} seconds.")
        print(f"Output saved entirely to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video through the Pure-Python GPU Edge Detector.")
    parser.add_argument('--input', type=str, required=True, help="Path to input video file (e.g. data/input.mp4)")
    parser.add_argument('--output', type=str, default="out/edge_cupy_video.mp4", help="Path to save output video (default: out/edge_cupy_video.mp4)")
    parser.add_argument('--base_t', type=float, default=0.6, help="Threshold multiplier (default: 0.6)")
    
    args = parser.parse_args()
    process_video(args.input, args.output, base_t=args.base_t)
