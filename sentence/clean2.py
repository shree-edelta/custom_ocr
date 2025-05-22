import cv2
import numpy as np

# Load image
img = cv2.imread("../images/name13.jpg")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 10
)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
no_lines = cv2.subtract(thresh, detected_lines)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_lines, connectivity=8)
min_area = 250  # area threshold for keeping components (adjust as needed)

mask = np.zeros(no_lines.shape, dtype="uint8")
for i in range(1, num_labels):  # Skip background (label 0)
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        mask[labels == i] = 255

# Step 3: Invert to get white background
final = cv2.bitwise_not(mask)

# Step 4: Crop whitespace
coords = cv2.findNonZero(255 - final)
x, y, w, h = cv2.boundingRect(coords)
cropped = final[y:y+h, x:x+w]

# Save cleaned image
cv2.imwrite("cleaned_image2.png", cropped)
cv2.imshow('Final Cleaned Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
