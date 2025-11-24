import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = np.random.randn(25, 25).astype(np.float32)
        x = (x - x.min()) / (x.max() - x.min()) * 256         
        y = (x - 128) / 10
        return torch.tensor(x).unsqueeze(0), torch.tensor(y).unsqueeze(0)


def train_model(model, train_loader, val_loader=None,
                epochs=5, lr=1e-3, device='cpu', save_path='best_model.pth'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

   
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_loss += criterion(model(x), y).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print("  -> Saved best model")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
            torch.save(model.state_dict(), save_path)  

    return model


def test_model(model, test_loader, device='cpu', weight_path='best_model.pth'):
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"no:"+weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            print(x.shape,type(x))
            total_loss += criterion(model(x), y).item()
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":

    save_path = "/data/wangkexin/changecode/Convolution/Pixel_Offset_Extractor/checkpoint/best_model.pth"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 
    train_set = DummyDataset(8000)
    val_set   = DummyDataset(200)
    test_set  = DummyDataset(200)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=16, shuffle=False)

    # 
    model = SimpleCNN()

    # 
    train_model(model, train_loader, val_loader, epochs=100, save_path = save_path, device=device)
    # 
    test_model(model, test_loader, weight_path=save_path, device=device)