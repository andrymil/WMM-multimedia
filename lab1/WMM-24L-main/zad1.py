import cv2
import numpy as np
import matplotlib.pyplot as plt
import time 
from scripts import cv_imshow, calcPSNR

def ex_1():
    path_org = './obrazy_testowe/color/boat2_col.png'
    path_inoise = './obrazy_testowe/color_inoise2/boat2_col_inoise.png'
    path_noise = './obrazy_testowe/color_noise/boat2_col_noise.png'

    img_org = cv2.imread(path_org, cv2.IMREAD_UNCHANGED)
    img_inoise = cv2.imread(path_inoise, cv2.IMREAD_UNCHANGED)
    img_noise = cv2.imread(path_noise, cv2.IMREAD_UNCHANGED)

    mask = [3, 5, 7]

    for x in mask:
        # Gausa
        inoise_g = cv2.GaussianBlur(img_inoise, (x, x), 0)
        noise_g = cv2.GaussianBlur(img_noise, (x, x), 0)

        # Medianowy
        inoise_m = cv2.medianBlur(img_inoise, x)
        noise_m = cv2.medianBlur(img_noise, x)

        # Wyswietl
        cv_imshow(inoise_g, f'Gauss inoise - {x}')
        cv_imshow(noise_g, f'Gauss noise - {x}')

        cv_imshow(inoise_g, f'Median inoise - {x}')
        cv_imshow(noise_g, f'Median noise - {x}')
        # time.sleep(3)

        #psnr
        psnr_inoise_g = round(calcPSNR(img_org,inoise_g),3)
        psnr_noise_g = round(calcPSNR(img_org,noise_g),3)
    
        psnr_inoise_m = round(calcPSNR(img_org,inoise_m),3)
        psnr_noise_m = round(calcPSNR(img_org,noise_m),3)

        print(f'PSNR Gauss inoise - mask {x} | {psnr_inoise_g}')
        print(f'PSNR Gauss noise - mask {x} | {psnr_noise_g}')
        print(f'PSNR Median inoise - mask {x} | {psnr_inoise_m}')
        print(f'PSNR Median noise - mask {x} | {psnr_noise_m}')

        #zapis
        cv2.imwrite(f'./output/inoise_g_{x}.png', inoise_g)
        cv2.imwrite(f'./output/noise_g_{x}.png', noise_g)
        cv2.imwrite(f'./output/inoise_m_{x}.png', inoise_m)
        cv2.imwrite(f'./output/noise_m_{x}.png', noise_m)


ex_1()