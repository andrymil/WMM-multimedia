import cv2
import numpy as np
def cv_imshow(img, img_title="image"):
    # cv2.namedWindow(img_title, cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_NORMAL
   
    ##### przeskalowanie obrazu z rzeczywistymi wartościami pikseli, żeby jedną funkcją wywietlać obrazy różnych typów
    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img/255
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(1)  ### oczekiwanie przez bardzo krótki czas - okno się wyświetli, ale program się nie zablokuje, tylko będzie kontynuowany

def calcPSNR(img1, img2):
    imax = 255.**2  ### zakładana wartość pikseli z przedziału [0, 255]
    ##### w różnicy obrazów istotne są wartości ujemne, dlatego img1 konwertowany do typu np.float64 (liczby rzeczywiste) aby nie ograniczać wyniku do przedziału [0, 255]
    mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size  ### img1.size - liczba elementów w img1, ==img1.shape[0]*img1.shape[1] dla obrazów mono, ==img1.shape[0]*img1.shape[1]*img1.shape[2] dla obrazów barwnych
    return 10.0*np.log10(imax/mse)