import json
import os
import math
import numpy as np

def calculate_average_rmse(image_folder_path, grouth_truth_path, predicted_path, confidence_threshold=0.5):
    # Đọc ground truth
    with open(grouth_truth_path, 'r') as f:
        gt_data = json.load(f)
    gt_dict = {item['id']: item['kps'] for item in gt_data}
    
    # Đọc predicted
    with open(predicted_path, 'r') as f:
        pred_data = json.load(f)
    pred_dict = {item['id']: item['kps'] for item in pred_data}
    
    squared_errors = []
    
    for img_id, gt_kps in gt_dict.items():
        if img_id not in pred_dict:
            continue
        
        pred_kps = pred_dict[img_id]
        
        for gt_point, pred_point in zip(gt_kps, pred_kps):
            x_gt, y_gt = gt_point
            x_pred, y_pred, conf = pred_point
            
            # Bỏ qua nếu không đủ confidence
            if conf < confidence_threshold:
                continue
            
            # Tính bình phương sai số Euclidean
            sq_err = (x_pred - x_gt)**2 + (y_pred - y_gt)**2
            squared_errors.append(sq_err)
    
    # Tính RMSE trung bình trên toàn bộ dataset
    if not squared_errors:
        return 0.0
    
    mse = np.mean(squared_errors)
    rmse = math.sqrt(mse)
    return rmse

def calculate_mean_rmse_per_image(image_folder_path, grouth_truth_path, predicted_path, confidence_threshold=0.5):
    # Đọc ground truth
    with open(grouth_truth_path, 'r') as f:
        gt_data = json.load(f)
    gt_dict = {item['id']: item['kps'] for item in gt_data}
    
    # Đọc predicted
    with open(predicted_path, 'r') as f:
        pred_data = json.load(f)
    pred_dict = {item['id']: item['kps'] for item in pred_data}

    rmse_results = {}
    rmse_results_norm = {}
    image_rmses = []
    image_rmses_norm = []

    for img_id, gt_kps in gt_dict.items():
        if img_id not in pred_dict:
            continue
        
        pred_kps = pred_dict[img_id]
        squared_errors = []
        
        for gt_point, pred_point in zip(gt_kps, pred_kps):
            x_gt, y_gt = gt_point
            x_pred, y_pred, conf = pred_point
            
            if conf < confidence_threshold:
                continue
            
            sq_err = (x_pred - x_gt)**2 + (y_pred - y_gt)**2
            squared_errors.append(sq_err)
        
        if squared_errors:
            mse = np.mean(squared_errors)
            rmse = math.sqrt(mse)
            rmse_results[img_id] = rmse
            image_rmses.append(rmse)

            # Normalize rmse
            x0, y0 = gt_kps[0]
            x3, y3 = gt_kps[3]
            L = math.sqrt((x3 - x0) ** 2 + (y3 - y0) ** 2)
            rmse_results_norm[img_id] = rmse / L if L > 0 else rmse
            image_rmses_norm.append(rmse_results_norm[img_id])

    if not image_rmses:
        return 0.0

    return np.mean(image_rmses), rmse_results, np.mean(image_rmses_norm) if image_rmses_norm else 0.0, rmse_results_norm

# Example usage
# image_folder_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\images_out"
# ground_truth_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\ground_truth.json"
# predicted_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\predicted.json"

# mean_rmse, rmse_results, mean_rmse_norm, rmse_results_norm = calculate_mean_rmse_per_image(image_folder_path, ground_truth_path, predicted_path, confidence_threshold=0.5)
# print("Mean RMSE per image:", mean_rmse)
# print("RMSE per image:", rmse_results)
# print("Mean RMSE per image (normalized):", mean_rmse_norm)
# print("RMSE per image (normalized):", rmse_results_norm)

# average_mse = calculate_average_rmse(image_folder_path, ground_truth_path, predicted_path, confidence_threshold=0.5)
# print("Average RMSE:", average_mse)