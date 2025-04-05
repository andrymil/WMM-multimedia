import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def cv_imshow(img, img_title="image"):
    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    elif img.dtype == np.int16:
        img_ = img * 128
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(1)


def calc_entropy(hist):
    pdf = hist / hist.sum()
    entropy = -sum([x * np.log2(x) for x in pdf if x != 0])
    return entropy


def get_histogram_and_entropy(img, size=255):
    hist = cv2.calcHist([img], [0], None, [size], [0, size]).flatten()
    entropy = calc_entropy(hist)
    return hist, entropy


def calc_diff_img(img):
    img_tmp1 = img[:, 1:]
    img_tmp2 = img[:, :-1]

    img_diff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0, dtype=cv2.CV_16S)
    img_diff_0 = cv2.addWeighted(img[:, 0], 1, 0, 0, -127, dtype=cv2.CV_16S)
    img_diff = np.hstack((img_diff_0, img_diff))

    return img_diff


def dwt(img):
    """
    Bardzo prosta i podstawowa implementacja, nie uwzględniająca efektywnych metod obliczania DWT
    i dopuszczająca pewne niedokładności.
    """
    maskL = np.array([0.02674875741080976, -0.01686411844287795, -0.07822326652898785, 0.2668641184428723,
        0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287795, 0.02674875741080976])
    maskH = np.array([0.09127176311424948, -0.05754352622849957, -0.5912717631142470, 1.115087052456994,
        -0.5912717631142470, -0.05754352622849957, 0.09127176311424948])

    bandLL = cv2.sepFilter2D(img,         -1, maskL, maskL)[::2, ::2]
    bandLH = cv2.sepFilter2D(img, cv2.CV_16S, maskL, maskH)[::2, ::2]
    bandHL = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskL)[::2, ::2]
    bandHH = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskH)[::2, ::2]

    return bandLL, bandLH, bandHL, bandHH


def calc_bitrate(img):
    cv2.imwrite("compressed_input.png", img)

    height, width = img.shape
    num_pixels = height * width

    file_size_bytes = os.path.getsize("compressed_input.png")

    bitrate = (file_size_bytes * 8) / num_pixels

    return bitrate


def plot_histograms(hist_input, hist_diff):
    plt.figure()
    plt.title("Histogram obrazu oryginalnego")
    plt.plot(hist_input)
    plt.xlabel("Wartość piksela")
    plt.ylabel("Liczba pikseli")
    plt.grid(True)
    plt.savefig("hist_original.png")
    plt.show()

    plt.figure()
    plt.title("Histogram obrazu różnicowego")
    plt.plot(np.arange(-255, 256, 1), hist_diff)
    plt.xlabel("Różnica pikseli")
    plt.ylabel("Liczba pikseli")
    plt.grid(True)
    plt.savefig("hist_diff.png")
    plt.show()


def plot_histograms_dwt(ll, lh, hl, hh):
    (hist_ll, entropy_ll) = ll
    (hist_lh, entropy_lh) = lh
    (hist_hl, entropy_hl) = hl
    (hist_hh, entropy_hh) = hh

    fig = plt.figure()
    fig.set_figheight(fig.get_figheight()*2)
    fig.set_figwidth(fig.get_figwidth()*2)
    plt.subplot(2, 2, 1)
    plt.plot(hist_ll, color="blue")
    plt.title(f"LL - {entropy_ll:.4f}")
    plt.xlim([0, 255])
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(-255, 256, 1), hist_lh, color="red")
    plt.title(f"LH - {entropy_lh:.4f}")
    plt.xlim([-255, 255])
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(-255, 256, 1), hist_hl, color="red")
    plt.title(f"HL - {entropy_hl:.4f}")
    plt.xlim([-255, 255])
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(-255, 256, 1), hist_hh, color="red")
    plt.title(f"HH - {entropy_hh:.4f}")
    plt.xlim([-255, 255])
    plt.tight_layout()
    plt.savefig("hist_dwt.png")
    plt.show()
    plt.close('all')


def save_dwt_bands(ll, lh, hl, hh):
    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(ll, cmap='gray')
    axes[0, 0].set_title("LL")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(lh, cmap='gray')
    axes[0, 1].set_title("LH")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(hl, cmap='gray')
    axes[1, 0].set_title("HL")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(hh, cmap='gray')
    axes[1, 1].set_title("HH")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('dwt_pasma.png')


def plot_comparison(data):
    labels = ["H(oryg)", "H(diff)", "H(LL)", "H(LH)", "H(HL)", "H(HH)", "Przepływność"]

    plt.figure()
    plt.title("Porównanie entropii")
    bars = plt.bar(labels, data)
    plt.bar_label(bars, fmt=lambda x: f'{x:.3f}')
    plt.ylabel("Entropia")
    plt.savefig("entropy_comparison.png")
    plt.show()


img = cv2.imread("original.png", cv2.IMREAD_UNCHANGED)

img_diff = calc_diff_img(img)

ll, lh, hl, hh = dwt(img)

cv_imshow(img, "Obraz oryginalny")
cv_imshow(img_diff, "Obraz roznicowy")
cv_imshow(ll, "LL")
cv_imshow(cv2.multiply(lh, 2), "LH")
cv_imshow(cv2.multiply(hl, 2), "HL")
cv_imshow(cv2.multiply(hh, 2), "HH")
cv2.waitKey(0)
cv2.destroyAllWindows()

hist_input, entropy_input = get_histogram_and_entropy(img)
hist_diff, entropy_diff = get_histogram_and_entropy((img_diff+255).astype(np.uint16), 511)

hist_ll = cv2.calcHist([ll], [0], None, [256], [0, 256]).flatten()
hist_lh = cv2.calcHist([(lh+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
hist_hl = cv2.calcHist([(hl+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
hist_hh = cv2.calcHist([(hh+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()

H_ll = calc_entropy(hist_ll)
H_lh = calc_entropy(hist_lh)
H_hl = calc_entropy(hist_hl)
H_hh = calc_entropy(hist_hh)

diff_shifted = (img_diff + 255).astype(np.uint16)
diff_normalized = cv2.normalize(diff_shifted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("diff.png", diff_normalized)

plot_histograms(hist_input, hist_diff)
plot_histograms_dwt((hist_ll, H_ll), (hist_lh, H_lh), (hist_hl, H_hl), (hist_hh, H_hh))

save_dwt_bands(ll, lh, hl, hh)

bitrate = calc_bitrate(img)

plot_comparison([entropy_input, entropy_diff, H_ll, H_lh, H_hl, H_hh, bitrate])

print("Entropia obrazu oryginalnego:", entropy_input)
print("Entropia obrazu różnicowego:", entropy_diff)
print(f"H(LL) = {H_ll:.4f} \nH(LH) = {H_lh:.4f} \nH(HL) = {H_hl:.4f} \nH(HH) = {H_hh:.4f} \nH_śr = {(H_ll+H_lh+H_hl+H_hh)/4:.4f}")
print(f"Przepływność PNG (oryginał): {bitrate:.4f} bpp")
