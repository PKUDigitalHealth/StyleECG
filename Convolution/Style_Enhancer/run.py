import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from PIL import Image

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

def generate_style_enhancer(styecg, styall, save_path, device):

    model = DualInputCNN()

    weight_path = save_path

    device = device

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    wriecg = []
    for ind in range(len(styecg)):
        x1 = np.array(styall[ind])
        x2 = np.array(styecg[ind])
        bgray = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY).astype(int)
        fimg = cv2.resize(x2, dsize = bgray.shape, interpolation=cv2.INTER_NEAREST).astype(int)
        
        fimg = fimg.transpose(2,0,1)
        
        
        a = fimg[0]*bgray/255
        b = fimg[1]*bgray/255
        c = fimg[2]*bgray/255
        
        
        f0 = fimg[0].reshape(1,1,32,32)
        f1 = fimg[1].reshape(1,1,32,32)
        f2 = fimg[2].reshape(1,1,32,32)
        bg = bgray.reshape(1,1,32,32)
        
        a1 = model(torch.tensor(f0, dtype=torch.float32, device=device), torch.tensor(bg, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(32,32)
        b1 = model(torch.tensor(f1, dtype=torch.float32, device=device), torch.tensor(bg, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(32,32)
        c1 = model(torch.tensor(f2, dtype=torch.float32, device=device), torch.tensor(bg, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(32,32)
        
        
        wriecg_path = np.stack([a1,b1,c1],axis=0)
     
        wriecg.append(Image.fromarray(wriecg_path.transpose(1, 2, 0).astype(np.uint8)))
    return wriecg
            