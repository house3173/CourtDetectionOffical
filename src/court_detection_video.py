import argparse
import os
import cv2
import time
import torch
from court_resnet50 import *
from ultralytics import YOLO

def process_frame(frame, model_detect, model_keypoints, device, input_keypoints_size, NUM_KEYPOINTS):
    # detect court with YOLO
    results = model_detect.predict(source=frame, conf=0.25, imgsz=640, save=False, verbose=False)
    boxes_court = results[0].boxes
    if len(boxes_court) == 0:
        # No court detected
        cv2.putText(frame, "No court detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    else:
        box_court = boxes_court[0].xyxy.cpu().numpy()
        confidence = boxes_court[0].conf.cpu().numpy()

        # detect keypoints
        # convert frame to temporary image array for preprocessing
        img_tensor, orig_size = preprocess_image(frame, input_keypoints_size)

        # predict keypoints
        coords = predict_keypoints(model_keypoints, img_tensor, orig_size, device)

        # draw keypoints
        for i, (x, y) in enumerate(coords):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # draw court bounding box
        cv2.rectangle(frame, (int(box_court[0][0] - 15), int(box_court[0][1] - 5)), 
                      (int(box_court[0][2] + 15), int(box_court[0][3] + 5)), (0, 0, 255), 2)

        return frame

def main():
    parser = argparse.ArgumentParser(description="Court keypoint detection inference on video")
    parser.add_argument("--model_detect_path", type=str, default="C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\models\\court_detection_yolov11n.pt", help="Path to model .pth file")
    parser.add_argument("--model_keypoints_path", type=str, default="C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\models\\court_keypoints_detection_resnet50.pth", help="Path to model .pth file")
    parser.add_argument("--video_path", type=str, default="C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\images\\test_video_5m.mp4", help="Path to input video")
    parser.add_argument("--output_path", type=str, default="C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\outputs\\test_video_5m_output.mp4", help="Path to output video")
    parser.add_argument("--input_keypoints_size", type=int, default=512, help="Model input size (default: 512)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    NUM_KEYPOINTS = 14

    # load models
    model_detect = YOLO(args.model_detect_path)
    model_keypoints = ResNetKeypointRegressor(num_keypoints=NUM_KEYPOINTS, pretrained=False).to(device)
    ok = load_checkpoint(model_keypoints, args.model_keypoints_path, device)
    if not ok:
        print("Warning: checkpoint could not be loaded cleanly.")

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    print(f"Processing {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, model_detect, model_keypoints, device, args.input_keypoints_size, NUM_KEYPOINTS)
        out.write(processed_frame)

    cap.release()
    out.release()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Video processing completed. Output saved to {args.output_path}")
    print(f"Total time: {elapsed_time:.2f} seconds, FPS: {frame_count / elapsed_time:.2f}")

if __name__ == '__main__':
    main()
