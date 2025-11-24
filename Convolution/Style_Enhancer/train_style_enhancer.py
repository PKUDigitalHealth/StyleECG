import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DualInputCNN(nn.Module):
    def __init__(self):
        super(DualInputCNN, self).__init__()
   
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x1, x2):
        # x1, x2: (B, 1, 32, 32)
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        f = torch.cat([f1, f2], dim=1)  # (B, 128, 32, 32)
        f = self.fusion(f)
        out = self.decoder(f)  # (B, 1, 32, 32)
        return out


class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        x1 = np.random.randn(25, 25).astype(np.float32)
        x1 = (x1 - x1.min()) / (x1.max() - x1.min()) * 256       
        
        x2 = np.random.randn(25, 25).astype(np.float32)
        x2 = (x2 - x2.min()) / (x2.max() - x2.min()) * 256    
        
   
        y = x1*x2/250
        return torch.tensor(x1).unsqueeze(0), torch.tensor(x2).unsqueeze(0), torch.tensor(y).unsqueeze(0)


def train_model(model, dataloader, epochs=5, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x1, x2)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def test_model(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
    
            pred = model(x1, x2)
            loss = criterion(pred, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Test MSE Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    train_set = DummyDataset(size=8000)
    test_set = DummyDataset(size=100)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    lr=1e-4

    model = DualInputCNN()

  
    train_model(model, train_loader, epochs=100,lr = lr, device=device)


    test_model(model, test_loader, device=device)


    torch.save(model.state_dict(), "/data/wangkexin/changecode/Convolution/Style_Enhance/best_model.pth")