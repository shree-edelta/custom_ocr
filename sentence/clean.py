import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('../images/Screenshot 2025-05-15 at 5.57.59 PM.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to binarize the image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 35, 10)

# Morphological operation to remove horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Subtract the lines from the original thresholded image
no_lines = cv2.subtract(thresh, detected_lines)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_lines, connectivity=8)
min_area = 250  # area threshold for keeping components (adjust as needed)

mask = np.zeros(no_lines.shape, dtype="uint8")
for i in range(1, num_labels):  # Skip background (label 0)
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        mask[labels == i] = 255


# Invert back for better visualization
result = cv2.bitwise_not(mask)

# Optional: Crop white space
coords = cv2.findNonZero(255 - result)  # get non-white pixel coords
x, y, w, h = cv2.boundingRect(coords)
cropped = result[y:y+h, x:x+w]

# Save or display result
cv2.imwrite('clean_output.png', cropped)

# If using Jupyter or matplotlib:
# plt.imshow(cropped, cmap='gray')
# plt.title('Final Cleaned Image')
# plt.axis('off')
# plt.show()

# If using OpenCV:
cv2.imshow('Final Cleaned Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()