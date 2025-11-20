import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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

X_train = np.load('X_train_ngsim.npy')
y_train = np.load('y_train_ngsim.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                          batch_size=512, shuffle=True)

model = MotionPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(15):
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.6f}")

torch.save(model.state_dict(), 'motion_model_relative.pt')
print(" Model trained on relative motion (Δx, Δy) and saved.")
