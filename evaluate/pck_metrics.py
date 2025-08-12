import json
import os
import math
from PIL import Image
from PIL import ImageDraw

images_out_folder_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\images_out"

def calculate_pck(image_folder_path, ground_truth_path, predicted_path, alpha=0.02, confidence_threshold=0.5):
    # Đọc file ground truth
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)
    gt_dict = {item['id']: item['kps'] for item in gt_data}
    
    # Đọc file predicted
    with open(predicted_path, 'r') as f:
        pred_data = json.load(f)
    pred_dict = {item['id']: item['kps'] for item in pred_data}
    
    total_keypoints = []
    correct_keypoints = []

    for img_id, gt_kps in gt_dict.items():
        total_keypoints_images = 0
        correct_keypoints_images = 0

        if img_id not in pred_dict:
            continue
        
        pred_kps = pred_dict[img_id]

        # Lấy kích thước ảnh để tính chiều dài chuẩn L
        img_path = os.path.join(image_folder_path, img_id + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_folder_path, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue  # bỏ qua nếu ảnh không tồn tại

        '''
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        '''

        # Tính L là khoảng cách giữa keypoints[0] và keypoints[3] của ground truth
        x0, y0 = gt_kps[0]
        x3, y3 = gt_kps[3]
        L = math.sqrt((x3 - x0) ** 2 + (y3 - y0) ** 2)

        for gt_point, pred_point in zip(gt_kps, pred_kps):
            x_gt, y_gt = gt_point
            x_pred, y_pred, conf = pred_point
            
            if conf < confidence_threshold:
                continue  # bỏ qua nếu không đủ confidence
            
            dist = math.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
            # Determine color based on correctness
            if dist <= alpha * L:
                correct_keypoints_images += 1
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Red

            '''
            # Draw keypoint on image
            radius = 5
            draw.ellipse(
                [(x_pred - radius, y_pred - radius), (x_pred + radius, y_pred + radius)],
                fill=color,
                outline=color
            )
            '''

            total_keypoints_images += 1
        '''
        # Save the image with drawn keypoints
        out_img_path = os.path.join(images_out_folder_path, img_id + ".png")
        img.save(out_img_path)
        '''

        total_keypoints.append(total_keypoints_images)
        correct_keypoints.append(correct_keypoints_images)

    pck = sum(correct_keypoints) / sum(total_keypoints) if total_keypoints else 0.0
    pck_list = [c / t if t > 0 else 0.0 for c, t in zip(correct_keypoints, total_keypoints)]

    return pck, pck_list

# Ví dụ gọi hàm
# image_folder_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\images"
# ground_truth_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\ground_truth.json"
# predicted_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\predicted.json"

# pck_score, pck_list = calculate_pck(image_folder_path, ground_truth_path, predicted_path, alpha=0.02, confidence_threshold=0.5)
# print("PCK score:", pck_score)
# print("PCK list:", pck_list)