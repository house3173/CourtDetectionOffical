from oks_metrics import calculate_average_oks
from pck_metrics import calculate_pck
from rmse import calculate_average_rmse, calculate_mean_rmse_per_image
import os
import numpy as np
from typing import List, Optional, Tuple

def evaluate_court_detection(
    image_folder_path: str,
    ground_truth_path: str,
    predicted_path: str,
    alpha_pck: float = 0.02,
    sigmas: Optional[np.ndarray] = np.array([0.075, 0.075, 0.060, 0.060, 0.0725, 0.0725, 0.0575, 0.0575, 0.07, 0.07, 0.055, 0.055, 0.055, 0.04]),
    confidence_threshold: float = 0.5,
    skip_images_without_prediction: bool = False
):

    pck_average, pck_results = calculate_pck(image_folder_path, ground_truth_path, predicted_path, alpha=alpha_pck, confidence_threshold=confidence_threshold)
    oks_results, oks_average = calculate_average_oks(image_folder_path, ground_truth_path, predicted_path, sigmas=sigmas, confidence_threshold=confidence_threshold, skip_images_without_prediction=skip_images_without_prediction)
    rmse_all = calculate_average_rmse(image_folder_path, ground_truth_path, predicted_path, confidence_threshold=confidence_threshold)
    rmse_average, rmse_results, rmse_average_norm, rmse_results_norm = calculate_mean_rmse_per_image(image_folder_path, ground_truth_path, predicted_path, confidence_threshold=confidence_threshold)

    return {
        "PCK": pck_average,
        "OKS": oks_average,
        "RMSE": rmse_all,
        "RMSE_average": rmse_average,
        "RMSE_average_norm": rmse_average_norm
    }

# Evaluation metrics for court detection
# image_folder_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\images"
# ground_truth_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\ground_truth.json"
# predicted_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\predicted.json"

# evaluate_results = evaluate_court_detection(image_folder_path, ground_truth_path, predicted_path)
# print(evaluate_results)