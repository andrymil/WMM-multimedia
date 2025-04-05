import matplotlib.pyplot as plt
import numpy as np
import cv2

def ex_2():
    path_org = './obrazy_testowe/color/boat2_col.png'
    img_org = cv2.imread(path_org, cv2.IMREAD_UNCHANGED)

    # Przestrzen YCBRC
    img_ycbcr = cv2.cvtColor(img_org, cv2.COLOR_BGR2YCrCb)
    img_ycbcr_org = img_ycbcr.copy()

    # Wyrownanie
    img_ycbcr[...,0] = cv2.equalizeHist(img_ycbcr[...,0])

    # powrot do RGB
    img_rgb = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2BGR)

    # zapisz
    cv2.imwrite('./output/eq_hist.png', img_rgb)

    plt.figure()
    hist_org_full = plt.hist(img_org.flatten(), 256, range=[0.0, 255.0])
    plt.title('Original histogram, every channel')
    plt.savefig('./output/org_hist_full.png')
    plt.figure()
    hist_org_y = plt.hist(img_ycbcr_org[...,0].flatten(), 256, range=[0.0, 255.0])
    plt.title('Original histogram, last channel')
    plt.savefig('./output/org_hist_last.png')
    plt.figure()
    hist_eq_full = plt.hist(img_rgb.flatten(), 256, range=[0.0, 255.0])
    plt.title('equalized histogram, every channel')
    plt.savefig('./output/eq_hist_full.png')
    plt.figure()
    hist_eq_y = plt.hist(img_ycbcr[..., 0].flatten(), 256, range=[0.0, 255.0])
    plt.title('equalized histogram, last channel')
    plt.savefig('./output/eq_hist_last.png')









ex_2()