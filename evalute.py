import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
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
    Evaluates the trained model on the 30% held-out test set.
    """
    files_needed = [
        'ngsim_test_30.csv', 
        'motion_model_relative.pt',
        'scaler_X.pkl',
        'scaler_y.pkl'
    ]
    
    if not all(os.path.exists(f) for f in files_needed):
        print("Error: Missing required files.")
        print("Please run 'preprocessing.py' and 'train.py' first.")
        return
        
    print("Loading 30% test set 'ngsim_test_30.csv'...")
    df_test = pd.read_csv('ngsim_test_30.csv')

    columns_needed = ['Vehicle_ID', 'Frame_ID', 'Global_X', 'Global_Y', 'v_Vel', 'v_Acc', 'Lane_ID']
    
    if not all(col in df_test.columns for col in columns_needed):
        print("Error: The 'ngsim_test_30.csv' file is missing one of the required columns.")
        print(f"Needed: {columns_needed}")
        return
        
    df_test = df_test[columns_needed].copy()
    
    df_test.columns = ['vehicle_id', 'frame_id', 'x_pos', 'y_pos', 'velocity', 'acceleration', 'lane']
    
    df_test = df_test.sort_values(['vehicle_id', 'frame_id']).reset_index(drop=True)

    print("Calculating ground truth motion for test set...")
    
    df_test['next_x'] = df_test.groupby('vehicle_id')['x_pos'].shift(-1)
    df_test['next_y'] = df_test.groupby('vehicle_id')['y_pos'].shift(-1)
    
    df_test['dx_true'] = df_test['next_x'] - df_test['x_pos']
    df_test['dy_true'] = df_test['next_y'] - df_test['y_pos']
    
    df_test = df_test.dropna(subset=['dx_true', 'dy_true'])
    
    print(f"Test data after removing NaN: {len(df_test)} rows")
    
    print("\nValidating and cleaning test data...")
    
    MAX_DISPLACEMENT = 10.0
    
    displacement_magnitude = np.sqrt(df_test['dx_true']**2 + df_test['dy_true']**2)
    outliers = displacement_magnitude > MAX_DISPLACEMENT
    
    print(f"Found {outliers.sum()} outliers (displacement > {MAX_DISPLACEMENT}m)")
    print(f"Outlier percentage: {100*outliers.sum()/len(df_test):.2f}%")
    
    if outliers.sum() > 0:
        print(f"Max displacement before cleaning: {displacement_magnitude.max():.2f} meters")
        print(f"Removing outliers...")
        df_test = df_test[~outliers].copy()
        print(f"Test data after outlier removal: {len(df_test)} rows")

    features = ['x_pos', 'y_pos', 'velocity', 'acceleration', 'lane']
    
    print("\nLoading scalers and trained model...")
    
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    model = MotionPredictor(input_dim=5, output_dim=2)
    model.load_state_dict(torch.load('motion_model_relative.pt', map_location='cpu'))
    model.eval()

    test_features_scaled = scaler_X.transform(df_test[features].values)

    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(test_features_scaled)).numpy()

    preds_unscaled = scaler_y.inverse_transform(preds_scaled)

    df_test['pred_dx'] = preds_unscaled[:, 0]
    df_test['pred_dy'] = preds_unscaled[:, 1]

    mae_dx = mean_absolute_error(df_test['dx_true'], df_test['pred_dx'])
    mae_dy = mean_absolute_error(df_test['dy_true'], df_test['pred_dy'])
    rmse = np.sqrt(mean_squared_error(df_test[['dx_true', 'dy_true']], df_test[['pred_dx', 'pred_dy']]))
    
    mean_displacement = np.sqrt(df_test['dx_true']**2 + df_test['dy_true']**2).mean()
    relative_error = (rmse / mean_displacement) * 100 if mean_displacement > 0 else 0
    
    print("\n" + "="*60)
    print("         Model Performance on 30% Test Set")
    print("="*60)
    print(f"📊 MAE (Δx):        {mae_dx:.4f} meters")
    print(f"📊 MAE (Δy):        {mae_dy:.4f} meters")
    print(f"📊 Overall RMSE:    {rmse:.4f} meters")
    print(f"📊 Relative Error:  {relative_error:.2f}%")
    print("="*60 + "\n")

    print("Ground Truth Statistics:")
    print(f"  dx range:  [{df_test['dx_true'].min():.4f}, {df_test['dx_true'].max():.4f}] meters")
    print(f"  dy range:  [{df_test['dy_true'].min():.4f}, {df_test['dy_true'].max():.4f}] meters")
    print(f"  dx mean:   {df_test['dx_true'].mean():.4f}, std: {df_test['dx_true'].std():.4f}")
    print(f"  dy mean:   {df_test['dy_true'].mean():.4f}, std: {df_test['dy_true'].std():.4f}")
    print(f"  Avg displacement: {mean_displacement:.4f} meters")
    
    print("\nPrediction Statistics:")
    print(f"  pred_dx range: [{df_test['pred_dx'].min():.4f}, {df_test['pred_dx'].max():.4f}] meters")
    print(f"  pred_dy range: [{df_test['pred_dy'].min():.4f}, {df_test['pred_dy'].max():.4f}] meters")
    print(f"  pred_dx mean:  {df_test['pred_dx'].mean():.4f}, std: {df_test['pred_dx'].std():.4f}")
    print(f"  pred_dy mean:  {df_test['pred_dy'].mean():.4f}, std: {df_test['pred_dy'].std():.4f}\n")

    df_test['pred_x_next'] = df_test['x_pos'] + df_test['pred_dx']
    df_test['pred_y_next'] = df_test['y_pos'] + df_test['pred_dy']
    df_test['true_x_next'] = df_test['x_pos'] + df_test['dx_true']
    df_test['true_y_next'] = df_test['y_pos'] + df_test['dy_true']

    vehicle_counts = df_test['vehicle_id'].value_counts()
    
    if not vehicle_counts.empty:
        suitable_vehicles = vehicle_counts[vehicle_counts >= 50]
        
        if len(suitable_vehicles) > 0:
            sample_vehicle_id = suitable_vehicles.index[0]
        else:
            sample_vehicle_id = vehicle_counts.index[0]
        
        sample_vehicle = df_test[df_test['vehicle_id'] == sample_vehicle_id].copy()
        sample_vehicle = sample_vehicle.sort_index()
        
        if len(sample_vehicle) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            ax1 = axes[0, 0]
            ax1.plot(sample_vehicle['x_pos'], sample_vehicle['y_pos'], 
                    'b-o', label='Current Position', linewidth=2, markersize=4)
            ax1.plot(sample_vehicle['true_x_next'], sample_vehicle['true_y_next'], 
                    'g-s', label='True Next Position', linewidth=2, markersize=4, alpha=0.7)
            ax1.plot(sample_vehicle['pred_x_next'], sample_vehicle['pred_y_next'], 
                    'r--^', label='Predicted Next Position', linewidth=1.5, markersize=4, alpha=0.7)
            
            ax1.set_xlabel("X Position (meters)", fontsize=11)
            ax1.set_ylabel("Y Position (meters)", fontsize=11)
            ax1.legend(fontsize=9)
            ax1.set_title(f"Trajectory Comparison - Vehicle {sample_vehicle_id}", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[0, 1]
            n_points = min(200, len(sample_vehicle))
            sample_subset = sample_vehicle.iloc[:n_points]
            
            ax2.scatter(sample_subset['dx_true'], sample_subset['dy_true'], 
                       c='green', label='True Δ', alpha=0.5, s=30)
            ax2.scatter(sample_subset['pred_dx'], sample_subset['pred_dy'], 
                       c='red', label='Predicted Δ', alpha=0.5, s=30, marker='^')
            
            ax2.set_xlabel("Δx (meters)", fontsize=11)
            ax2.set_ylabel("Δy (meters)", fontsize=11)
            ax2.legend(fontsize=9)
            ax2.set_title("Displacement Vector Comparison", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            
            ax3 = axes[1, 0]
            sample_vehicle['error'] = np.sqrt((sample_vehicle['pred_dx'] - sample_vehicle['dx_true'])**2 + 
                                             (sample_vehicle['pred_dy'] - sample_vehicle['dy_true'])**2)
            ax3.plot(range(len(sample_vehicle)), sample_vehicle['error'], 'purple', linewidth=1.5)
            ax3.axhline(y=rmse, color='r', linestyle='--', label=f'Overall RMSE: {rmse:.3f}m')
            ax3.set_xlabel("Time Step", fontsize=11)
            ax3.set_ylabel("Prediction Error (meters)", fontsize=11)
            ax3.set_title("Prediction Error Over Time", fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            all_errors = np.sqrt((df_test['pred_dx'] - df_test['dx_true'])**2 + 
                                (df_test['pred_dy'] - df_test['dy_true'])**2)
            ax4.hist(all_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.axvline(x=rmse, color='r', linestyle='--', linewidth=2, label=f'RMSE: {rmse:.3f}m')
            ax4.axvline(x=np.median(all_errors), color='green', linestyle='--', linewidth=2, 
                       label=f'Median: {np.median(all_errors):.3f}m')
            ax4.set_xlabel("Prediction Error (meters)", fontsize=11)
            ax4.set_ylabel("Frequency", fontsize=11)
            ax4.set_title("Error Distribution (All Test Data)", fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('trajectory_evaluation.png', dpi=150, bbox_inches='tight')
            print("✅ Saved comprehensive visualization to 'trajectory_evaluation.png'")
            plt.show()
        else:
            print(f"Could not find data for sample vehicle ID: {sample_vehicle_id}")
    else:
        print("No vehicle data found in the test set after processing.")

if __name__ == "__main__":
    evaluate_model()