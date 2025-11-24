import os
import clip
import cv2
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B/32", device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def extract_features(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
        return features

    def extract_features_from_grayscale_image(self, grayscale_image):
        grayscale_image = Image.fromarray(grayscale_image, mode='L')
        rgb_image = grayscale_image.convert('RGB')
        input_image = self.preprocess(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(input_image)
        return features


class CustomSegmentationDataset(Dataset):

    def __init__(self, con_dir, sty_dir, wri_dir, device, transform):
        self.device = device
    
        self.con_dir = con_dir
        self.sty_dir = sty_dir
        self.wri_dir = wri_dir
        self.transform = transform

    
  
        self.image_encoder = CLIPFeatureExtractor(model_name = "ViT-B/32")

    
        assert len(self.con_dir) == len(self.sty_dir) == len(self.wri_dir)

    def __len__(self):
        return len(self.con_dir)

    def __getitem__(self, idx):
     
        con_image = self.con_dir[idx].convert("RGB")
        if self.transform:
            con_image_t = self.transform(con_image)
       
        sty_image = self.sty_dir[idx].convert("RGB")
        if self.transform:
            sty_image_t = self.transform(sty_image)
      
        wri_image = self.wri_dir[idx].convert("RGB") 
        if self.transform:
            wri_image_t = self.transform(wri_image)

    
        con_encoder = self.image_encoder.extract_features(con_image)
      
        sty_encoder = self.image_encoder.extract_features(sty_image)
 
        wri_encoder = self.image_encoder.extract_features(wri_image)

        bimg = cv2.cvtColor(np.array(sty_image),cv2.COLOR_RGB2BGR)
        bgray = cv2.cvtColor(bimg, cv2.COLOR_BGR2GRAY).astype(int)
        goffset = (bgray - 128)/10
        offsetLim = (np.floor(goffset) + 13).astype(int)
     
        wrinkles_encoder = self.image_encoder.extract_features_from_grayscale_image(offsetLim)
        return con_encoder, sty_encoder, wri_encoder, wrinkles_encoder, con_image_t, sty_image_t, wri_image_t, str(idx) + ".png", offsetLim
