# OBRAZ 11
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv2_imshow(img, img_title="image"):
    # cv2.namedWindow(img_title, cv2.WINDOW_AUTOSIZE) # cv2.WINDOW_NORMAL

    ##### przeskalowanie obrazu z rzeczywistymi wartościami pikseli, żeby jedną funkcją wywietlać obrazy różnych typów
    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img/255
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(0)  ### oczekiwanie przez bardzo krótki czas - okno się wyświetli, ale program się nie zablokuje, tylko będzie kontynuowany
    cv2.destroyAllWindows()


def imshow(img, img_title="image"):  ### 'opakowanie' na cv2_imshow(), żeby 'uzgodnić' parametry wywołania
    cv2_imshow(img)


def printi(img, img_title="image"):
    print(f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}")


image = cv2.imread('obrazy/color/baboon_col.png', cv2.IMREAD_UNCHANGED)
image_noise = cv2.imread("obrazy/color_inoise1/baboon_col_inoise.png", cv2.IMREAD_UNCHANGED)
blur_img = cv2.blur(image_noise, (3, 3))
gblur_img = cv2.GaussianBlur(image_noise, (5, 5), 0)
median_img = cv2.medianBlur(image_noise, 3)
printi(image_noise, "image_noise")
cv2.imshow("image", image)
cv2.imshow("image_noise", image_noise)
cv2.imshow("blur_img", blur_img)
cv2.imshow("gblur_img", gblur_img)
cv2.imshow("median_img", median_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
