import cv2
import numpy as np

def ex_3():
    path_org = './obrazy_testowe/color/boat2_col.png'
    img_org = cv2.imread(path_org, cv2.IMREAD_UNCHANGED)

    # wstepne wygladzenie obrazu filtrem Gaussa
    img_org = cv2.GaussianBlur(img_org, (1, 1), 0)
    laplace = cv2.Laplacian(img_org, cv2.CV_64F)

    # correct error (converting to uint8)
    img_org = cv2.convertScaleAbs(img_org)
    laplace = cv2.convertScaleAbs(laplace)

    ws = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    neg_ws = [ -1*w for w in ws]

    for w in neg_ws:
        sharp = cv2.addWeighted(img_org, 1, laplace, w, 0)
        cv2.imwrite(f'./output/sharp_{abs(w)}.png', sharp)

ex_3()