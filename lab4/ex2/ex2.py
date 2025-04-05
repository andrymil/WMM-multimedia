import cv2
import matplotlib.pyplot as plt


def plot_histogram(channel, title, color='black'):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.title(title)
    plt.xlabel("Poziom jasności (0-255)")
    plt.ylabel("Liczba pikseli")
    plt.xlim([0, 256])


image = cv2.imread('../original.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Konwersja do przestrzeni YCrCb
image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(image_ycrcb)

# Wyrównanie histogramu jasności
y_eq = cv2.equalizeHist(y)

# Konwersja z powrotem do RGB
image_eq_ycrcb = cv2.merge((y_eq, cr, cb))
image_eq = cv2.cvtColor(image_eq_ycrcb, cv2.COLOR_YCrCb2RGB)

cv2.imwrite("image_equalized.png", cv2.cvtColor(image_eq, cv2.COLOR_RGB2BGR))

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Obraz oryginalny
axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title("Oryginalny obraz")
axs[0, 0].axis('off')

# Obraz po wyrównaniu histogramu
axs[0, 1].imshow(image_eq)
axs[0, 1].set_title("Po wyrównaniu histogramu")
axs[0, 1].axis('off')

# Histogram jasności oryginału
plt.subplot(2, 2, 3)
plot_histogram(y, "Histogram jasności - oryginał")

# Histogram jasności po wyrównaniu
plt.subplot(2, 2, 4)
plot_histogram(y_eq, "Histogram jasności - wyrównany")

plt.tight_layout()
plt.savefig("histogram_comparison.png", dpi=300)
plt.show()
