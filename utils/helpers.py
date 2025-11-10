import torch
import numpy as np
import logging
import os
import json
from datetime import datetime

def setup_logging(log_dir='./logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'sportiq_{datetime.now().strftime("%Y%m%d")}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    return epoch, metrics

def calculate_metrics(predictions, targets):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        mse = mean_squared_error(targets, predictions, multioutput='raw_values')
        mae = mean_absolute_error(targets, predictions, multioutput='raw_values')
        r2 = r2_score(targets, predictions, multioutput='raw_values')
    else:
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
    
    return {'mse': mse, 'mae': mae, 'r2': r2}

def create_directories():
    dirs = ['./models', './data', './logs', './results', './exports']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def normalize_keypoints(keypoints, image_shape):
    height, width = image_shape[:2]
    normalized = {}
    
    for joint_name, joint_data in keypoints.items():
        normalized[joint_name] = {
            'x': joint_data['x'] / width,
            'y': joint_data['y'] / height,
            'confidence': joint_data['confidence']
        }
    
    return normalized

def denormalize_keypoints(keypoints, image_shape):
    height, width = image_shape[:2]
    denormalized = {}
    
    for joint_name, joint_data in keypoints.items():
        denormalized[joint_name] = {
            'x': joint_data['x'] * width,
            'y': joint_data['y'] * height,
            'confidence': joint_data['confidence']
        }
    
    return denormalized

def smooth_poses(poses, window_size=5):
    if len(poses) < window_size:
        return poses
    
    smoothed_poses = []
    
    for i in range(len(poses)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(poses), i + window_size // 2 + 1)
        
        window_poses = poses[start_idx:end_idx]
        smoothed_pose = {}
        
        for joint in poses[0].keys():
            x_values = [p[joint]['x'] for p in window_poses if p[joint]['confidence'] > 0.3]
            y_values = [p[joint]['y'] for p in window_poses if p[joint]['confidence'] > 0.3]
            
            if x_values and y_values:
                smoothed_pose[joint] = {
                    'x': np.mean(x_values),
                    'y': np.mean(y_values),
                    'confidence': np.mean([p[joint]['confidence'] for p in window_poses])
                }
            else:
                smoothed_pose[joint] = poses[i][joint]
        
        smoothed_poses.append(smoothed_pose)
    
    return smoothed_poses