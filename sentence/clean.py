import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/a01-000u-00-06.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 35, 10)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

no_lines = cv2.subtract(thresh, detected_lines)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_lines, connectivity=8)
min_area = 250 

mask = np.zeros(no_lines.shape, dtype="uint8")
for i in range(1, num_labels): 
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        mask[labels == i] = 255


result = cv2.bitwise_not(mask)

coords = cv2.findNonZero(255 - result) 
x, y, w, h = cv2.boundingRect(coords)
cropped = result[y:y+h, x:x+w]

cv2.imwrite('clean_output2.png', cropped)

cv2.imshow('Final Cleaned Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Read image
# img = cv2.imread("../images/20250520_130200.jpg", 0)

# # Threshold image
# _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# # Define a horizontal kernel (e.g., 40 pixels wide)
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

# # Detect horizontal lines
# detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

# # Subtract lines from original
# removed = cv2.subtract(binary, detected_lines)

# # Invert back
# result = 255 - removed

# # Save or show result
# cv2.imwrite("line_removed.png", result)
# cv2.imshow("Line Removed", result)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()