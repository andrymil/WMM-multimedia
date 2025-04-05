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


def get_histogram_and_entropy(img, size=256):
    hist = cv2.calcHist([img], [0], None, [size], [0, size]).flatten()

    entropy = calc_entropy(hist)
    return hist, entropy


def show_rgb_channels(r_channel, g_channel, b_channel):
    zeros = np.zeros_like(r_channel)

    r_image = np.stack([r_channel, zeros, zeros], axis=2).astype(np.uint8)
    g_image = np.stack([zeros, g_channel, zeros], axis=2).astype(np.uint8)
    b_image = np.stack([zeros, zeros, b_channel], axis=2).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(r_image)
    axes[0].set_title("Red Channel")
    axes[1].imshow(g_image)
    axes[1].set_title("Green Channel")
    axes[2].imshow(b_image)
    axes[2].set_title("Blue Channel")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("rgb_channels.png")
    plt.show()


def plot_rgb_histograms(hist_R, hist_G, hist_B):
    plt.plot(hist_R, color="red")
    plt.plot(hist_G, color="green")
    plt.plot(hist_B, color="blue")
    plt.title("hist RGB")
    plt.xlim([0, 256])
    plt.legend(["Red", "Green", "Blue"])
    plt.savefig("rgb_hist.png")
    plt.show()


def show_yuv_channels(img_yuv):
    Y = img_yuv[:, :, 0]
    U = img_yuv[:, :, 1]
    V = img_yuv[:, :, 2]

    u_vis = np.full_like(img_yuv, 127)
    v_vis = np.full_like(img_yuv, 127)
    u_vis[:, :, 1] = U
    v_vis[:, :, 2] = V

    u_rgb = cv2.cvtColor(u_vis, cv2.COLOR_YUV2RGB)
    v_rgb = cv2.cvtColor(v_vis, cv2.COLOR_YUV2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(Y, cmap="gray")
    axes[0].set_title("Y Channel")
    axes[1].imshow(u_rgb)
    axes[1].set_title("U Channel")
    axes[2].imshow(v_rgb)
    axes[2].set_title("V Channel")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("yuv_channels.png")
    plt.show()


def plot_yuv_histograms(hist_Y, hist_U, hist_V):
    plt.plot(hist_Y, color="gray")
    plt.plot(hist_U, color="red")
    plt.plot(hist_V, color="blue")
    plt.title("hist YUV")
    plt.xlim([0, 256])
    plt.legend(["Y", "U", "V"])
    plt.savefig("yuv_hist.png")
    plt.show()


def calc_mse_psnr(img1, img2):
    imax = 255.**2

    mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size
    psnr = 10.0*np.log10(imax/mse)
    return (mse, psnr)


def compare_compression_qualities(image, qualities):
    xx = []
    ym = []
    yp = []

    for quality in qualities:
        out_file_name = f"jpeg/out_image_q{quality:03d}.jpg"

        cv2.imwrite(out_file_name, image, (cv2.IMWRITE_JPEG_QUALITY, quality))

        image_compressed = cv2.imread(out_file_name, cv2.IMREAD_UNCHANGED)
        bitrate = 8*os.stat(out_file_name).st_size/(image.shape[0]*image.shape[1])
        mse, psnr = calc_mse_psnr(image, image_compressed)

        xx.append(bitrate)
        ym.append(mse)
        yp.append(psnr)

    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth()*2)
    plt.suptitle("Charakterystyki R-D")
    plt.subplot(1, 2, 1)
    plt.plot(xx, ym, "-.")
    plt.title("MSE(R)")
    plt.xlabel("bitrate")
    plt.ylabel("MSE", labelpad=0)
    plt.subplot(1, 2, 2)
    plt.plot(xx, yp, "-o")
    plt.title("PSNR(R)")
    plt.xlabel("bitrate")
    plt.ylabel("PSNR [dB]", labelpad=0)
    plt.savefig("quality_comparison.png")
    plt.show()


def show_jpeg_qualities(image, qualities):
    assert len(qualities) == 6, "The list must contain exactly 6 quality values."

    fig, axes = plt.subplots(3, 2, figsize=(10, 11))

    for i, q in enumerate(qualities):
        filename = f"jpeg/out_image_q{q:03d}.jpg"
        cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, q])
        img_rgb = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        bitrate = 8*os.stat(filename).st_size/(image.shape[0]*image.shape[1])

        col, row = divmod(i, 3)
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(f"Quality {q}, Bitrate: {bitrate:.2f}")
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("images_to_evaluate.png")
    plt.show()


image_col = cv2.imread("original.png")

image_R = image_col[:, :, 2]
image_G = image_col[:, :, 1]
image_B = image_col[:, :, 0]

hist_R, H_R = get_histogram_and_entropy(image_R)
hist_G, H_G = get_histogram_and_entropy(image_G)
hist_B, H_B = get_histogram_and_entropy(image_B)
print(f"H(R) = {H_R:.4f} \nH(G) = {H_G:.4f} \nH(B) = {H_B:.4f} \nH_śr = {(H_R+H_G+H_B)/3:.4f}")

show_rgb_channels(image_R, image_G, image_B)

plot_rgb_histograms(hist_R, hist_G, hist_B)


image_yuv = cv2.cvtColor(image_col, cv2.COLOR_BGR2YCrCb)

image_Y = image_yuv[:, :, 0]
image_U = image_yuv[:, :, 1]
image_V = image_yuv[:, :, 2]

hist_Y, H_Y = get_histogram_and_entropy(image_Y)
hist_U, H_U = get_histogram_and_entropy(image_U)
hist_V, H_V = get_histogram_and_entropy(image_V)
print(f"H(Y) = {H_Y:.4f} \nH(U) = {H_U:.4f} \nH(V) = {H_V:.4f} \nH_śr = {(H_Y+H_U+H_V)/3:.4f}")

show_yuv_channels(image_yuv)

plot_yuv_histograms(hist_Y, hist_U, hist_V)

qualities = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95, 97, 98, 99, 100]
compare_compression_qualities(image_col, qualities)

qualities_to_evaluate = [1, 5, 15, 25, 50, 95]
show_jpeg_qualities(image_col, qualities_to_evaluate)

cv2.imwrite("compressed_input.png", image_col)
bitrate = 8*os.stat("compressed_input.png").st_size/(image_col.shape[0]*image_col.shape[1])
print(f"Bitrate for PNG: {bitrate:.2f} bpp")
