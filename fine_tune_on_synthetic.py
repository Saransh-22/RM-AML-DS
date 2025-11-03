import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

class MotionPredictor(nn.Module):
    def __init__(self, input_dim=5, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def evaluate_model():
    """
    Evaluates the relative motion model on the synthetic dataset.
    This FINAL version correctly scales BOTH inputs (X) and outputs (Y)
    using the original NGSIM scaler parameters.
    """

    files_needed = [
        'synthetic_traffic.csv', 
        'motion_model_relative.pt',
        'scaler_y_min.npy', 
        'scaler_y_scale.npy',
        'scaler_X_min.npy',
        'scaler_X_scale.npy'
    ]
    
    if not all(os.path.exists(f) for f in files_needed):
        print("Error: Missing required scaler files.")
        print("Please ensure 'Preprocessing.py' has been run successfully to create all .npy files.")
        return

    df_custom = pd.read_csv('synthetic_traffic.csv')

    df_target = df_custom.groupby('vehicle_id')[['x_pos', 'y_pos']].shift(-1)
    df_custom['dx_true'] = df_target['x_pos'] - df_custom['x_pos']
    df_custom['dy_true'] = df_target['y_pos'] - df_custom['y_pos']
    df_custom = df_custom.dropna()

    features = ['x_pos', 'y_pos', 'velocity', 'acceleration', 'lane']
    
    scaler_X_min = np.load('scaler_X_min.npy')
    scaler_X_scale = np.load('scaler_X_scale.npy')

    custom_features_scaled = (df_custom[features].values - scaler_X_min) * scaler_X_scale


    model = MotionPredictor(input_dim=5, output_dim=2)
    model.load_state_dict(torch.load('motion_model_relative.pt'))
    model.eval()

    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(custom_features_scaled)).numpy()

    scaler_y_min = np.load('scaler_y_min.npy')
    scaler_y_scale = np.load('scaler_y_scale.npy')

    preds_unscaled = (preds_scaled * scaler_y_scale) + scaler_y_min
    
    df_custom['pred_dx'] = preds_unscaled[:, 0]
    df_custom['pred_dy'] = preds_unscaled[:, 1]

    mae_dx = mean_absolute_error(df_custom['dx_true'], df_custom['pred_dx'])
    mae_dy = mean_absolute_error(df_custom['dy_true'], df_custom['pred_dy'])
    rmse = np.sqrt(mean_squared_error(df_custom[['dx_true', 'dy_true']], df_custom[['pred_dx', 'pred_dy']]))
    print(f"📊 Final MAE (Δx): {mae_dx:.4f}, MAE (Δy): {mae_dy:.4f}, RMSE: {rmse:.4f}")

    df_custom['pred_x_next'] = df_custom['x_pos'] + df_custom['pred_dx']
    df_custom['pred_y_next'] = df_custom['y_pos'] + df_custom['pred_dy']

    plt.figure(figsize=(10, 6)) 
    sample_vehicle_id = df_custom['vehicle_id'].value_counts().index[0]
    sample_vehicle = df_custom[df_custom['vehicle_id'] == sample_vehicle_id]
    
    plt.plot(sample_vehicle['x_pos'], sample_vehicle['y_pos'], 'b-', label='Actual Path', linewidth=2)
    plt.plot(sample_vehicle['pred_x_next'], sample_vehicle['pred_y_next'], 'r--', label='Predicted Next Path')
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    plt.title("Actual vs. Correctly Predicted Next Trajectory")
    plt.grid(True)
    
    plt.ylim(0, 15)
    
    plt.show()

if __name__ == "__main__":
    evaluate_model()

