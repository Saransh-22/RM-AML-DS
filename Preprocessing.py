import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_data(filepath="NGSIM_vehicle_trajectory.csv"):
    """
    Preprocesses the NGSIM dataset to predict relative motion (dx, dy).
    This FINAL version saves all 4 required scaler files.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print("Please download the NGSIM dataset and place it in the correct directory.")
        return

    print("Loading NGSIM data...")
    df = pd.read_csv(filepath)

    columns_needed = ['Vehicle_ID', 'Global_X', 'Global_Y', 'v_Vel', 'v_Acc', 'Lane_ID']
    df = df[columns_needed]
    df.columns = ['vehicle_id', 'x_pos', 'y_pos', 'velocity', 'acceleration', 'lane']

    print("Calculating relative motion (Δx, Δy)...")
    df_target = df.groupby('vehicle_id')[['x_pos', 'y_pos']].shift(-1)
    df_target.columns = ['next_x', 'next_y']

    df['dx'] = df_target['next_x'] - df['x_pos']
    df['dy'] = df_target['next_y'] - df['y_pos']

    df = df.dropna()

    features = ['x_pos', 'y_pos', 'velocity', 'acceleration', 'lane']
    target = ['dx', 'dy']

    print("Scaling data...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[target])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    print("Saving preprocessed data and scalers...")
    np.save('X_train_ngsim.npy', X_train)
    np.save('y_train_ngsim.npy', y_train)
    np.save('X_test_ngsim.npy', X_test)
    np.save('y_test__ngsim.npy', y_test) 
    
    np.save('scaler_X_min.npy', scaler_X.min_)
    np.save('scaler_X_scale.npy', scaler_X.scale_)
    
    np.save('scaler_y_min.npy', scaler_y.min_)
    np.save('scaler_y_scale.npy', scaler_y.scale_)

    print("✅ Relative motion preprocessing complete. All 4 scaler files saved.")

if __name__ == "__main__":
    preprocess_data()

