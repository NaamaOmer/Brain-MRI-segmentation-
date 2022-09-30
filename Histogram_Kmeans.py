import cv2
import matplotlib.pyplot as plt
import numpy as np

# Naama Omer

img = cv2.imread('12.tif')
mask = cv2.imread('12_mask.tif')

# alternative way to find histogram of an image
plt.hist(img.ravel(), 256, [0, 256])
plt.savefig("histogram")
plt.show()


img = cv2.imread('12.tif')
mask = cv2.imread('12_mask.tif')
path = '12_mask.tif'
colormap1 = cv2.imread(path)
colormap1=cv2.cvtColor(colormap1, cv2.COLOR_BGR2RGB)
plt.imshow(colormap1)
# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.figure()
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        img[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")
plt.show()


# Change color to RGB (from BGR)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1, 3))
# Convert to float type
pixel_vals = np.float32(pixel_vals)
# the below line of code defines the criteria for the algorithm to stop running,
# which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
# becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
# then perform k-means clustering wit h number of clusters defined as 3
# also random centres are initially choosed for k-means clustering
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape(image.shape)

plt.imshow(segmented_image)
plt.show()

plt.imshow(mask)
plt.show()
