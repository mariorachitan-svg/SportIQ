import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class VisualizationEngine:
    def __init__(self):
        self.color_palette = {
            'team1': (255, 0, 0),
            'team2': (0, 0, 255),
            'keypoints': (0, 255, 0),
            'skeleton': (255, 255, 0),
            'text': (255, 255, 255)
        }
    
    def draw_pose_on_frame(self, frame, pose, confidence_threshold=0.3):
        frame_copy = frame.copy()
        
        skeleton_connections = [
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        
        height, width = frame.shape[:2]
        
        for connection in skeleton_connections:
            joint1, joint2 = connection
            
            if (joint1 in pose and joint2 in pose and
                pose[joint1]['confidence'] > confidence_threshold and
                pose[joint2]['confidence'] > confidence_threshold):
                
                x1 = int(pose[joint1]['x'] * width)
                y1 = int(pose[joint1]['y'] * height)
                x2 = int(pose[joint2]['x'] * width)
                y2 = int(pose[joint2]['y'] * height)
                
                cv2.line(frame_copy, (x1, y1), (x2, y2), self.color_palette['skeleton'], 2)
        
        for joint_name, joint_data in pose.items():
            if joint_data['confidence'] > confidence_threshold:
                x = int(joint_data['x'] * width)
                y = int(joint_data['y'] * height)
                
                cv2.circle(frame_copy, (x, y), 4, self.color_palette['keypoints'], -1)
                cv2.putText(frame_copy, joint_name, (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color_palette['text'], 1)
        
        return frame_copy
    
    def create_performance_dashboard(self, performance_data):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Movement Efficiency', 'Fatigue Index', 
                          'Flexibility Score', 'Power Output'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=performance_data.get('movement_efficiency', 0) * 100,
                title={'text': "Movement Efficiency"},
                domain={'row': 0, 'column': 0},
                gauge={'axis': {'range': [0, 100]}}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=performance_data.get('fatigue_index', 0) * 100,
                title={'text': "Fatigue Index"},
                domain={'row': 0, 'column': 1},
                gauge={'axis': {'range': [0, 100]}}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=performance_data.get('flexibility_score', 0) * 100,
                title={'text': "Flexibility Score"},
                domain={'row': 1, 'column': 0},
                gauge={'axis': {'range': [0, 100]}}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=performance_data.get('power_output', 0),
                title={'text': "Power Output"},
                domain={'row': 1, 'column': 1},
                gauge={'axis': {'range': [0, 1000]}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Athlete Performance Dashboard")
        return fig
    
    def plot_motion_trajectory(self, trajectories, player_ids=None):
        fig = go.Figure()
        
        if player_ids is None:
            player_ids = list(trajectories.keys())
        
        for player_id in player_ids:
            if player_id in trajectories:
                traj = trajectories[player_id]
                if traj:
                    x_coords = [point[0] for point in traj]
                    y_coords = [point[1] for point in traj]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='lines+markers',
                        name=f'Player {player_id}',
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
        
        fig.update_layout(
            title='Player Movement Trajectories',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            showlegend=True
        )
        
        return fig
    
    def create_biomechanics_report(self, biomechanics_data):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Joint Angles Over Time', 'Velocity Profile',
                          'Acceleration Analysis', 'Injury Risk Assessment'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        if 'joint_angles' in biomechanics_data:
            angles_data = biomechanics_data['joint_angles']
            for joint, angles in angles_data.items():
                fig.add_trace(
                    go.Scatter(y=angles, name=joint, mode='lines'),
                    row=1, col=1
                )
        
        if 'velocities' in biomechanics_data:
            velocities_data = biomechanics_data['velocities']
            for joint, vels in velocities_data.items():
                fig.add_trace(
                    go.Scatter(y=vels, name=joint, mode='lines'),
                    row=1, col=2
                )
        
        fig.update_layout(height=800, title_text="Biomechanics Analysis Report")
        return fig
    
    def generate_heatmap(self, trajectories, field_dimensions=(1.0, 1.0), grid_size=20):
        heatmap_data = np.zeros((grid_size, grid_size))
        
        for trajectory in trajectories.values():
            for point in trajectory:
                x, y = point
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    heatmap_data[grid_y, grid_x] += 1
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Player Position Heatmap',
            xaxis_title='Field Width',
            yaxis_title='Field Length'
        )
        
        return fig