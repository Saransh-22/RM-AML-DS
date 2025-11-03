import numpy as np
import pandas as pd
import random

def generate_synthetic_traffic(num_vehicles=50, timesteps=300, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    data = []
    event_types = ['normal', 'brake', 'lane_change', 'near_collision']

    for vid in range(num_vehicles):
        vehicle_id = f"V{vid+1}"
        lane = random.randint(1, 3)
        x, y = np.random.rand() * 100, lane * 3.5
        velocity = np.random.uniform(10, 30)
        acceleration = 0

        for t in range(timesteps):
            time = round(t * 0.1, 1)
            accel_change = np.random.uniform(-1, 1)
            acceleration = np.clip(accel_change, -3, 2)
            velocity = max(0, velocity + acceleration * 0.1)
            x += velocity * 0.1
            y = lane * 3.5

            event = 'normal'
            if random.random() < 0.01:
                event = random.choice(event_types)
                if event == 'brake':
                    velocity = max(0, velocity - 5)
                elif event == 'lane_change':
                    lane = random.randint(1, 3)
                    y = lane * 3.5
                elif event == 'near_collision':
                    velocity = max(0, velocity - np.random.uniform(5, 10))

            data.append([time, vehicle_id, x, y, velocity, acceleration, lane, event])

    df = pd.DataFrame(data, columns=['time', 'vehicle_id', 'x_pos', 'y_pos', 'velocity', 'acceleration', 'lane', 'event_type'])
    df.to_csv('synthetic_traffic.csv', index=False)
    print("Synthetic dataset saved as 'synthetic_traffic.csv'")
    return df

df = generate_synthetic_traffic()
