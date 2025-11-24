import cv2
import numpy as np
 
def newwrinkles(style_path, new_image ,output_path):
    bimg = cv2.imread(style_path)

    bgray = cv2.cvtColor(bimg, cv2.COLOR_BGR2GRAY).astype(int)

    fimg0 = cv2.imread(new_image)
    #fimg0 = cv2.cvtColor(np.array(new_image),cv2.COLOR_RGB2BGR)
    fimg = cv2.resize(fimg0, dsize = bgray.shape, interpolation=cv2.INTER_NEAREST).astype(int)

    ######print(fimg.shape, bgray.shape)
    #print(fimg.transpose(2,0,1).shape)
    #print(type(fimg), type(bgray))
    newarr = fimg.transpose(2,0,1)*bgray/255
    newarr = newarr.transpose((1,2,0)).astype(np.uint8)
    data = cv2.resize(newarr,dsize=(1100,850))
    cv2.imwrite(output_path, data)



#newwrinkles1(style_path,cont_path,output_path)