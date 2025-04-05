import cv2
import numpy as np
import matplotlib.pyplot as plt


def sharpen_laplacian(img_f, weight):
    laplace = cv2.Laplacian(img_f, cv2.CV_32F)

    laplace_vis = cv2.convertScaleAbs(laplace)
    cv2.imwrite("laplace.png", laplace_vis)

    result = cv2.addWeighted(src1=img_f, alpha=1.0, src2=laplace, beta=-weight, gamma=0)
    return np.clip(result, 0, 255).astype(np.uint8)


img = cv2.imread("../original.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

blurred = cv2.GaussianBlur(img_rgb, (3, 3), 0)

blurred_f = blurred.astype(np.float32)

weights = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

fig, axs = plt.subplots(4, 2, figsize=(10, 15))

# Obraz oryginalny
axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title("Oryginalny wygładzony")
axs[0, 0].axis('off')

# Wyostrzanie dla każdej wagi
for i, w in enumerate(weights):
    row = (i + 1) // 2
    col = (i + 1) % 2
    sharpened = sharpen_laplacian(blurred_f, w)
    axs[row, col].imshow(sharpened)
    axs[row, col].set_title(f"W = {w}")
    axs[row, col].axis('off')

plt.tight_layout()
plt.savefig("wyostrzanie_laplace.png", dpi=300)
plt.show()
