# Vehicle Motion Prediction using Neural Networks

A deep learning project that predicts vehicle motion trajectories using data from the NGSIM (Next Generation Simulation) dataset. The model uses a neural network to predict relative displacement (Δx, Δy) based on vehicle state features.

## 📋 Project Overview

This project implements a motion prediction model that:
- Preprocesses NGSIM vehicle trajectory data
- Trains a neural network to predict relative vehicle motion
- Evaluates model performance on test data
- Visualizes predicted vs. actual trajectories

## 🗂️ Project Structure

```
.
├── Preprocessing.py          # Data preprocessing and train/test split
├── Train.py                  # Neural network training script
├── evalute.py               # Model evaluation on test set
├── motion_model_relative.pt  # Trained model weights
├── X_train_ngsim.npy        # Preprocessed training features
├── y_train_ngsim.npy        # Preprocessed training targets
├── scaler_*.npy             # Scaler parameters for normalization
├── ngsim_train_70.csv       # Training data (70% split)
├── ngsim_test_30.csv        # Test data (30% split)
└── NGSIM_Vehicle_trajectory.csv  # Original NGSIM dataset
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Installation

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### Dataset

Download the NGSIM vehicle trajectory dataset from the official source:

**[NGSIM Dataset Download Link](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)**

Alternative sources:
- [NGSIM Community Website](https://www.fhwa.dot.gov/publications/research/operations/07030/)
- [Kaggle NGSIM Dataset](https://www.kaggle.com/search?q=ngsim)

Place the downloaded CSV file as `NGSIM_Vehicle_trajectory.csv` in the project root directory.

## 📊 Usage

### 1. Data Preprocessing

Run the preprocessing script to prepare the data:

```bash
python Preprocessing.py
```

This will:
- Split the dataset into 70% training and 30% testing
- Calculate relative motion (Δx, Δy) for each timestep
- Remove outliers and invalid data
- Normalize features using MinMaxScaler
- Save processed data and scaler parameters

### 2. Train the Model

Train the neural network:

```bash
python Train.py
```

This trains a feedforward neural network with:
- Input features: x_pos, y_pos, velocity, acceleration, lane
- Output: Δx, Δy (relative displacement)
- Architecture: 5 → 64 → 32 → 2 neurons
- Loss function: MSE (Mean Squared Error)
- Optimizer: Adam

### 3. Evaluate the Model

Evaluate model performance on the test set:

```bash
python evalute.py
```

This will:
- Load the test dataset
- Generate predictions using the trained model
- Calculate MAE and RMSE metrics
- Visualize predicted vs. actual trajectories

## 🧠 Model Architecture

```
MotionPredictor(
  (net): Sequential(
    (0): Linear(in_features=5, out_features=64)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=2)
  )
)
```

**Input Features (5):**
- x_pos: Global X position
- y_pos: Global Y position
- velocity: Vehicle velocity
- acceleration: Vehicle acceleration
- lane: Lane ID

**Output (2):**
- Δx: Change in X position (next timestep)
- Δy: Change in Y position (next timestep)

## 📈 Performance Metrics

The model is evaluated using:
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual displacements
- **RMSE (Root Mean Squared Error)**: Root mean squared error of position predictions

## 🔧 Configuration

Key hyperparameters in `Train.py`:
- Batch size: 512
- Learning rate: 0.001
- Epochs: 15
- Optimizer: Adam
- Loss function: MSELoss

## 📝 Data Format

The NGSIM dataset should contain the following columns:
- `Vehicle_ID`: Unique vehicle identifier
- `Frame_ID`: Frame number (timestep)
- `Global_X`: X position in global coordinates
- `Global_Y`: Y position in global coordinates
- `v_Vel`: Vehicle velocity
- `v_Acc`: Vehicle acceleration
- `Lane_ID`: Lane identifier

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- NGSIM dataset provided by the Federal Highway Administration (FHWA)
- Vehicle trajectory data collected on US Highway 101 and I-80

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note:** Make sure all CSV files are properly gitignored to avoid pushing large datasets to the repository.
