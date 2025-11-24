import os
import time
xpath_0 = "/data/wangkexin/ECGstyle/lunwen/physionet.org/files/ptb-xl/1.0.3/1/0/"
ximages_0 = os.listdir(xpath_0)
xpath_1 = "/data/wangkexin/ECGstyle/lunwen/physionet.org/files/ptb-xl/1.0.3/1/1/"
ximages_1 = os.listdir(xpath_1)
xpath_2 = "/data/wangkexin/ECGstyle/lunwen/physionet.org/files/ptb-xl/1.0.3/1/2/"
ximages_2 = os.listdir(xpath_2)
xpath_3 = "/data/wangkexin/ECGstyle/lunwen/physionet.org/files/ptb-xl/1.0.3/1/3/"
ximages_3 = os.listdir(xpath_3)
xpath_16 = "/data/wangkexin/ECGstyle/lunwen/physionet.org/files/ptb-xl/1.0.3/1/16/"
ximages_16 = os.listdir(xpath_16)
xpath_18 = "/data/wangkexin/ECGstyle/lunwen/physionet.org/files/ptb-xl/1.0.3/1/18/"
ximages_18 = os.listdir(xpath_18)
xpath_19 = "/data/wangkexin/ECGstyle/lunwen/physionet.org/files/ptb-xl/1.0.3/1/19/"
ximages_19 = os.listdir(xpath_19)


path_0 = "/data/wangkexin/ECGstyle/lunwen/output/1/0/"
images_0 = os.listdir(path_0)
path_1 = "/data/wangkexin/ECGstyle/lunwen/output/1/1/"
images_1 = os.listdir(path_1)
path_2 = "/data/wangkexin/ECGstyle/lunwen/output/1/2/"
images_2 = os.listdir(path_2)
path_3 = "/data/wangkexin/ECGstyle/lunwen/output/1/3/"
images_3 = os.listdir(path_3)
path_16 = "/data/wangkexin/ECGstyle/lunwen/output/1/16/"
images_16 = os.listdir(path_16)
path_18 = "/data/wangkexin/ECGstyle/lunwen/output/1/18"
images_18 = os.listdir(path_18)
path_19 = "/data/wangkexin/ECGstyle/lunwen/output/1/19"
images_19 = os.listdir(path_19)

print("0:",time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(path_0))),len(images_0)," / ",len(ximages_0)*3)
print("1:",time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(path_1))),len(images_1)," / ",len(ximages_1)*3)
print("2:",time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(path_2))),len(images_2)," / ",len(ximages_2)*3)
print("3:",time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(path_3))),len(images_3)," / ",len(ximages_3)*3)
print("16:",time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(path_16))),len(images_16)," / ",len(ximages_16)*3)
print("18:",time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(path_18))),len(images_18)," / ",len(ximages_18)*3)
print("19:",time.strftime('%H:%M:%S',time.localtime(os.path.getmtime(path_19))),len(images_19)," / ",len(ximages_19)*3)
#print(time.ctime(os.path.getmtime(path_4)),len(images_4)," / ",len(ximages_4)*3)
