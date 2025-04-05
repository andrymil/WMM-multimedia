import cv2
import numpy as np


def calcPSNR(img1, img2):
    imax = 255.0 ** 2
    mse = ((img1.astype(np.float64) - img2) ** 2).sum() / img1.size
    return 10.0 * np.log10(imax / mse)


original = cv2.imread("../original.png")
impulse_noise = cv2.imread("../impulse.png")
gauss_noise = cv2.imread("../gaussian.png")

kernel_sizes = [(3, 3), (5, 5), (7, 7)]

for ksize in kernel_sizes:
    print(f"\nMaska: {ksize}")
    save_dir = f"kernel_{ksize[0]}"

    # Szum gaussowski
    blur_gauss = cv2.GaussianBlur(gauss_noise, ksize, 0)
    median_gauss = cv2.medianBlur(gauss_noise, ksize[0])

    cv2.imwrite(f"{save_dir}/gauss_noise/gauss.png", blur_gauss)
    cv2.imwrite(f"{save_dir}/gauss_noise/median.png", median_gauss)

    psnr_blur_gauss = calcPSNR(original, blur_gauss)
    psnr_median_gauss = calcPSNR(original, median_gauss)

    print("szum gaussowski:")
    print(f"Gauss blur {psnr_blur_gauss:.2f} dB")
    print(f"Median blur {psnr_median_gauss:.2f} dB\n")

    # Szum impulsowy
    blur_impulse = cv2.GaussianBlur(impulse_noise, ksize, 0)
    median_impulse = cv2.medianBlur(impulse_noise, ksize[0])

    cv2.imwrite(f"{save_dir}/impulse_noise/gauss.png", blur_impulse)
    cv2.imwrite(f"{save_dir}/impulse_noise/median.png", median_impulse)

    psnr_blur_impulse = calcPSNR(original, blur_impulse)
    psnr_median_impulse = calcPSNR(original, median_impulse)

    print("szum impulsowy:")
    print(f"Gauss blur: {psnr_blur_impulse:.2f} dB")
    print(f"Median blur: {psnr_median_impulse:.2f} dB")

    # Wyświetlenie obrazów
    cv2.imshow("Original", original)
    cv2.imshow("Impulse noise", impulse_noise)
    cv2.imshow("Gauss noise", gauss_noise)

    cv2.imshow("Gauss noise, Gauss blur", blur_gauss)
    cv2.imshow("Gauss noise, Median blur", median_gauss)

    cv2.imshow("Impulse noise, Gauss blur", blur_impulse)
    cv2.imshow("Impulse noise, Median blur", median_impulse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
