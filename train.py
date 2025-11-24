import os
from typing import Dict
from PIL import Image

import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from model import UNet
from Scheduler import GradualWarmupScheduler
from load import CustomSegmentationDataset

def delete_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset

    con_dir = "/data/wangkexin/ECGstyle/data/train/con/"
    sty_dir = "/data/wangkexin/ECGstyle/data/train/sty/"
    tar_dir = "/data/wangkexin/ECGstyle/data/train/tar/"
    wri_dir = "/data/wangkexin/ECGstyle/data/train/wri/"

    print("load dataset")
    dataset = CustomSegmentationDataset(con_dir, sty_dir, tar_dir, wri_dir, device, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True)




    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for batch_idx, (con_encoder, sty_encoder, tar_encoder, wri_encoder, wrinkles_encoder, con_image_t, sty_image_t, tar_image_t, wri_image_t, index, offsetLim) in enumerate(dataloader):
    
                tar_image_t , wri_image_t = tar_image_t.to(device), wri_image_t.to(device)
                sty_encoder, wri_encoder = sty_encoder.to(device), wri_encoder.to(device)
                offsetLim = offsetLim.to(device)    
    
                # train
                optimizer.zero_grad()
                x_0 = tar_image_t.to(device)
            
                loss = trainer(x_0, wri_encoder.float(), sty_encoder.float()).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + str(loss) + "_.pt"))


def eval(modelConfig: Dict, model,split_file_con_path,split_file_sty_path,split_file_wri_path):
    # load model and evaluate
    with torch.no_grad():
        all_image = [Image.new('RGB', (32, 32), 'white') for _ in range(1024)]
        device = torch.device(modelConfig["device"])
        print("load dataset")
        dataset = CustomSegmentationDataset(split_file_con_path,split_file_sty_path,split_file_wri_path, device, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
        dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True)

      
        model.eval()
        sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        #delete_folder_contents("/data/wkx/w/DenoisingDiffusionProbabilityModel-ddpm--main/SampledImgs/")
        for con_encoder, sty_encoder, wri_encoder, wrinkles_encoder, con_image_t, sty_image_t, wri_image_t, index, offsetLim in tqdm(dataloader):
            wri_image_t, sty_image_t = wri_image_t.to(device), sty_image_t.to(device)
            sty_encoder, wri_encoder = sty_encoder.to(device), wri_encoder.to(device)
            offsetLim = offsetLim.to(device) 

            noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
           
            sampledImgs = sampler(noisyImage, wri_encoder.float(), sty_encoder.float(), wri_image_t.float(), sty_image_t.float())
            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            for ind, img in enumerate(sampledImgs):
                img = img.mul(255).add_(5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                im = Image.fromarray(np.uint8(img))
                #im.save(modelConfig["sampled_dir"] + index[ind])
           
                all_image[int(index[ind][:-4])] = im
            #save_image(sampledImgs, os.path.join(
             #   modelConfig["sampled_dir"],  index), nrow=modelConfig["nrow"])
            '''
            for ind, img in enumerate(con_image_t):
                img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            
                im = Image.fromarray(img)

                im.save("/home/wangkexin/ECGstyle/SampledImgs/ablation_no_first/" + index[ind])
            '''
        return all_image