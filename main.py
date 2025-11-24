from train import train, eval
import os, argparse
from split_combine import split_image, combine_tiles
from step2 import newwrinkles
from model import UNet
import torch
import random
import cv2
import numpy as np
from PIL import Image

def newwrinkle(style_path,cont_path,output_path):
    bimg = cv2.imread(style_path)
    bgray = cv2.cvtColor(bimg, cv2.COLOR_BGR2GRAY).astype(int)

    fimg0 = cv2.imread(cont_path)
 
    fimg = cv2.resize(fimg0, dsize = bgray.shape, interpolation=cv2.INTER_NEAREST).astype(int)

    Y,X,N = fimg.shape
    exfimg = np.zeros((Y+26,X+26,N))
    exfimg[13:-13, 13:-13, :] = fimg#
    for i in range(13):#
        exfimg[13:-13, i, :] = fimg[:,0,:]#
        exfimg[13:-13, -i-1, :] = fimg[:,-1,:]#
    for i in range(13):#
        exfimg[i, :, :] = exfimg[13,:,:]#
        exfimg[-i-1, :, :] = exfimg[-14,:,:]#
    
        
    Y,X,N = fimg.shape
    exfimg = np.zeros((Y+39,X+39,N))
    exfimg[26:-13, 26:-13, :] = fimg#
    for i in range(15):#
        exfimg[26:-13, i, :] = fimg[:,1,:]#
        exfimg[26:-13, i+13, :] = fimg[:,1,:]#
        exfimg[26:-13, -i-1, :] = fimg[:,-1,:]#
    for i in range(15):#
        exfimg[i, :, :] = exfimg[28,:,:]#
        exfimg[i+13, :, :] = exfimg[28,:,:]#
        exfimg[-i-1, :, :] = exfimg[-14,:,:]#
 
  
    goffset = (bgray - 128)/10
    offsetLim1 = (np.floor(goffset) + 13).astype(int)
    offsetLim2 = (np.ceil(goffset) + 13).astype(int)
    sep1 = (goffset - np.floor(goffset)).flatten()
    sep2 = (np.ceil(goffset) - goffset).flatten()
    XX, YY = np.meshgrid(range(exfimg.shape[0]-39),range(exfimg.shape[1]-39))
    XX1, YY1 = XX + offsetLim1, YY + offsetLim1#
    XX2, YY2 = XX + offsetLim2, YY + offsetLim2#


    c1 = exfimg[YY1.flatten(), XX1.flatten(), :]
    c2 = exfimg[YY2.flatten(), XX2.flatten(), :]
    
    p1 = np.where(sep1 == 0, c1[:,0], c2[:,0]*sep1.flatten() + c1[:,0]*sep2.flatten())
    p2 = np.where(sep1 == 0, c1[:,1], c2[:,1]*sep1.flatten() + c1[:,1]*sep2.flatten())
    p3 = np.where(sep1 == 0, c1[:,2], c2[:,2]*sep1.flatten() + c1[:,2]*sep2.flatten()) 

    newarr = np.array([p1.reshape(bgray.shape),p2.reshape(bgray.shape),p3.reshape(bgray.shape)])

    newarr = newarr.transpose((1,2,0)).astype(np.uint8)
    data = cv2.resize(newarr,dsize=(1100,850))

    cv2.imwrite(output_path, data)
    return output_path


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-device', type=int, required=False, default = 0)

    return parser

#There are a total of 4 addresses
#conall_path
#styall_path
#sampled_dir
#ckpt_path


if __name__ == '__main__':
    args = get_parser().parse_args()
    print("test")

    ecgall_path = "/data/wangkexin/changecode/con/"
    styall_path = "/data/wangkexin/changecode/sty/"
    disall_path = "/data/wangkexin/changecode/dis/" #The address of Distorted Patches
    sampled_dir = "/data/wangkexin/changecode/gen/" #Address for generating images
	
    if not os.path.exists(disall_path):
        os.makedirs(disall_path, exist_ok=True)
	
    if not os.path.exists(sampled_dir):
        os.makedirs(sampled_dir, exist_ok=True)
	
    ecgall_file = sorted(os.listdir(ecgall_path))
    styall_file = sorted(os.listdir(styall_path))  #The style images are read in order, so please keep the number of style images equal to the number of ECGs. 
	                                               #If you need to randomly read or the number of style images is not equal to ECG, please modify it yourself.

    for index, file in enumerate(ecgall_file):#The pictures are processed one by one. Read one by one, and then generate one by one
    
        print("index:", index, file)

        
        ecg_path = ecgall_path + ecgall_file[index]
        sty_path = styall_path + styall_file[index]#The style images are read in order, so please keep the number of style images equal to the number of ECGs. 
	                                               #If you need to randomly read or the number of style images is not equal to ECG, please modify it yourself.
        
        with Image.open(sty_path) as img1:
            resized_img = img1.resize((1100,1100)) #Change the resolution of the style image to 1100 * 1100 and overwrite the original image
            resized_img.save(sty_path)
			
			
        dis_path = newwrinkle(sty_path,ecg_path,sampled_dir + "w_" + file)
        
        ecgall = split_image(ecg_path, tile_size=(32, 32))
        styall = split_image(sty_path, tile_size=(32, 32))
		
		
		
        disall = split_image(dis_path, tile_size=(32, 32))

        modelConfig = {
            "state": "eval", # or eval or train
            "epoch": 200,
            "batch_size": 64,
            "T": 1000,
            "channel": 128,
            "channel_mult": [1, 2, 3, 4],
            "attn": [2],
            "num_res_blocks": 2,
            "dropout": 0.15,
            "lr": 1e-4,
            "multiplier": 2.,
            "beta_1": 1e-4,
            "beta_T": 0.02,
            "img_size": 32,
            "grad_clip": 1.,
            "device": "cuda:" + str(args.device), ### MAKE SURE YOU HAVE A GPU !!!
            "training_load_weight": None,
            "nrow": 8
        }
        ckpt_path = "/home/wangkexin/ECGstyle/ckpt/1/ckpt_9tensor(3.9829, device='cuda:0', grad_fn=<DivBackward0>)_.pt"
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
            num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(ckpt_path, map_location=modelConfig["device"])
        model.load_state_dict(ckpt)
        print("start")
       
        all_image = eval(modelConfig, model,ecgall,styall,disall)

        new_image = combine_tiles(all_image, sampled_dir + file, (32,32))
        new_image.save(sampled_dir + "z_" + file)
    
        newwrinkles(sty_path, new_image, sampled_dir + "z_" + file)

