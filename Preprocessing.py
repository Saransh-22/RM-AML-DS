import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def preprocess_data(full_filepath="NGSIM_vehicle_trajectory.csv"):
    """
    NEW WORKFLOW:
    1. Splits the entire NGSIM dataset into 70% train and 30% test CSVs.
    2. Processes ONLY the 70% train data to create training .npy files
       and the all-important scaler files.
    """
    if not os.path.exists(full_filepath):
        print(f"Error: File not found at {full_filepath}")
        return

    print(f"Loading full dataset from {full_filepath}...")
    df = pd.read_csv(full_filepath)
    
    print(f"Initial dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("Splitting data by Vehicle_ID (70% train, 30% test)...")
    vehicle_ids = df['Vehicle_ID'].unique()
    train_ids, test_ids = train_test_split(vehicle_ids, test_size=0.3, random_state=42)

    df_train = df[df['Vehicle_ID'].isin(train_ids)].copy()
    df_test = df[df['Vehicle_ID'].isin(test_ids)].copy()

    df_train.to_csv('ngsim_train_70.csv', index=False)
    df_test.to_csv('ngsim_test_30.csv', index=False)
    print(f" Saved 'ngsim_train_70.csv' ({len(df_train)} rows)")
    print(f" Saved 'ngsim_test_30.csv' ({len(df_test)} rows)")

    print("\nProcessing 'ngsim_train_70.csv' for training...")
    
    columns_needed = ['Vehicle_ID', 'Frame_ID', 'Global_X', 'Global_Y', 'v_Vel', 'v_Acc', 'Lane_ID']
    df_train = df_train[columns_needed].copy()
    df_train.columns = ['vehicle_id', 'frame_id', 'x_pos', 'y_pos', 'velocity', 'acceleration', 'lane']
    
    df_train = df_train.sort_values(['vehicle_id', 'frame_id']).reset_index(drop=True)

    print("Calculating relative motion (Δx, Δy) for training data...")
    
    df_train['next_x'] = df_train.groupby('vehicle_id')['x_pos'].shift(-1)
    df_train['next_y'] = df_train.groupby('vehicle_id')['y_pos'].shift(-1)
    
    df_train['dx'] = df_train['next_x'] - df_train['x_pos']
    df_train['dy'] = df_train['next_y'] - df_train['y_pos']
    
    df_train = df_train.dropna(subset=['dx', 'dy'])
    
    print(f"Training data after removing NaN: {len(df_train)} rows")
    
    print("\nValidating and cleaning data...")
    
    MAX_DISPLACEMENT = 10.0 
    
    displacement_magnitude = np.sqrt(df_train['dx']**2 + df_train['dy']**2)
    outliers = displacement_magnitude > MAX_DISPLACEMENT
    
    print(f"Found {outliers.sum()} outliers (displacement > {MAX_DISPLACEMENT}m)")
    print(f"Outlier percentage: {100*outliers.sum()/len(df_train):.2f}%")
    
    if outliers.sum() > 0:
        print(f"Max displacement before cleaning: {displacement_magnitude.max():.2f} meters")
        print(f"Removing outliers...")
        df_train = df_train[~outliers].copy()
        print(f"Training data after outlier removal: {len(df_train)} rows")
    
    print("\nTraining data statistics:")
    print(f"  dx range: [{df_train['dx'].min():.4f}, {df_train['dx'].max():.4f}] meters")
    print(f"  dy range: [{df_train['dy'].min():.4f}, {df_train['dy'].max():.4f}] meters")
    print(f"  dx mean: {df_train['dx'].mean():.4f}, std: {df_train['dx'].std():.4f}")
    print(f"  dy mean: {df_train['dy'].mean():.4f}, std: {df_train['dy'].std():.4f}")
    print(f"  velocity range: [{df_train['velocity'].min():.2f}, {df_train['velocity'].max():.2f}]")
    print(f"  acceleration range: [{df_train['acceleration'].min():.2f}, {df_train['acceleration'].max():.2f}]")

    features = ['x_pos', 'y_pos', 'velocity', 'acceleration', 'lane']
    target = ['dx', 'dy']

    print("\nFitting scalers ONLY on training data...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(df_train[features])
    y_train_scaled = scaler_y.fit_transform(df_train[target])

    print("Saving training data and scaler objects...")
    np.save('X_train_ngsim.npy', X_train_scaled)
    np.save('y_train_ngsim.npy', y_train_scaled)
    
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    np.save('scaler_X_min.npy', scaler_X.min_)
    np.save('scaler_X_scale.npy', scaler_X.scale_)
    np.save('scaler_X_data_min.npy', scaler_X.data_min_)
    np.save('scaler_X_data_max.npy', scaler_X.data_max_)
    
    np.save('scaler_y_min.npy', scaler_y.min_)
    np.save('scaler_y_scale.npy', scaler_y.scale_)
    np.save('scaler_y_data_min.npy', scaler_y.data_min_)
    np.save('scaler_y_data_max.npy', scaler_y.data_max_)

    print("\n Preprocessing complete. Training files and scalers are ready.")
    print(f"   Training samples: {len(X_train_scaled)}")
    print(f"   Number of unique vehicles: {df_train['vehicle_id'].nunique()}")

if __name__ == "__main__":
    preprocess_data()