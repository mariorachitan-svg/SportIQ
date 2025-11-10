import yaml
import os

class Config:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self):
        return {
            'model': {
                'pose_estimation': {
                    'input_size': [256, 256],
                    'num_keypoints': 17,
                    'confidence_threshold': 0.3
                },
                'action_recognition': {
                    'sequence_length': 16,
                    'num_actions': 10
                }
            },
            'processing': {
                'video': {
                    'target_fps': 30,
                    'max_frames': 1000
                },
                'tracking': {
                    'max_age': 30,
                    'min_hits': 3
                }
            },
            'analysis': {
                'biomechanics': {
                    'velocity_threshold': 0.5,
                    'acceleration_threshold': 0.8
                },
                'performance': {
                    'fatigue_threshold': 0.3,
                    'efficiency_threshold': 0.7
                }
            },
            'api': {
                'host': 'localhost',
                'port': 8000,
                'debug': True
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def update(self, key, value):
        keys = key.split('.')
        config_ref = self.config
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        config_ref[keys[-1]] = value
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f)