import argparse
import os
import cv2
import time
import torch
from court_resnet50 import *
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Court keypoint detection inference")
    parser.add_argument("--model_detect_path", type=str, default="C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\models\\court_detection_yolov11n.pt", help="Path to model .pth file")
    parser.add_argument("--model_keypoints_path", type=str, default="C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\models\\court_keypoints_detection_resnet50.pth", help="Path to model .pth file")
    parser.add_argument("--image_path", type=str, default="C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\images\\not_tennis_tennis.jpg", help="Path to input image")
    parser.add_argument("--input_keypoints_size", type=int, default=512, help="Model input size (default: 512)")
    parser.add_argument("--output_path", type=str, default="C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\outputs\\not_tennis_tennis_output.jpg", help="Path to output image")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    NUM_KEYPOINTS = 14

    # build model (do not download pretrained weights by default)
    model_detect = YOLO(args.model_detect_path)
    model_keypoints = ResNetKeypointRegressor(num_keypoints=NUM_KEYPOINTS, pretrained=False).to(device)
    ok = load_checkpoint(model_keypoints, args.model_keypoints_path, device)
    if not ok:
        print("\nWarning: checkpoint could not be loaded cleanly. The predictions may be invalid.")

    # load image and detect court with YOLO as prepare-step for keypoint detection
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    image = cv2.imread(args.image_path)

    start_time = time.time()

    results = model_detect.predict(source=image, conf=0.25, imgsz=640, save=False, verbose=False)
    boxes_court = results[0].boxes
    if len(boxes_court) == 0:
        print("No court detected in the image.")
        cv2.putText(image, "No court detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(args.output_path, image)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nInference time: {elapsed_time:.2f} seconds")

        return
    else:
        box_court = boxes_court[0].xyxy.cpu().numpy()
        confidence = boxes_court[0].conf.cpu().numpy()
        print(f"\nDetected court bounding box: {box_court[0]}, confidence: {confidence[0]}")

        # detect keypoints
        img_tensor, orig_size = preprocess_image(args.image_path, args.input_keypoints_size)

        # predict
        coords = predict_keypoints(model_keypoints, img_tensor, orig_size, device)

        # Print list of 14 (x,y) tuples
        print(f"\nDetected keypoints: {coords}")

        # Draw bbox - confidence and 14 keypoints, then save images
        for i, (x, y) in enumerate(coords):
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(image, str(i), (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw court bounding box
        cv2.rectangle(image, (int(box_court[0][0] - 15), int(box_court[0][1] - 5)), (int(box_court[0][2] + 15), int(box_court[0][3] + 5)), (0, 0, 255), 2)

        output_path = args.output_path
        cv2.imwrite(output_path, image)
        print(f"\nOutput image saved to: {output_path}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nInference time: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()
