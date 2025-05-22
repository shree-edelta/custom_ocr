# # Convert the image to grayscale

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# image = cv2.imread("images/Screenshot 2025-05-15 at 6.13.52 PM.png")

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply GaussianBlur to reduce noise and improve edge detection
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # Apply edge detection (Canny Edge Detection)
# edges = cv2.Canny(blurred, 50, 150)
# plt.imshow(edges, cmap='gray')
# plt.savefig("edges_canny_2.png")
# # Find contours in the edge-detected image
# # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # # Create a copy of the original image to draw contours on
# # contour_image = image.copy()

# # # Set width and height thresholds for horizontal line detection
# # width_threshold = gray.shape[1] * 0.6  # The line should cover at least 60% of the image width
# # height_threshold = 15  # The line should be relatively thin

# # # Initialize a list for storing detected horizontal lines
# # horizontal_lines = []

# # # Loop over all contours and filter horizontal ones
# # for contour in contours:
# #     x, y, w, h = cv2.boundingRect(contour)

# #     # Check if the contour is a horizontal line (wide and thin)
# #     if w > width_threshold and h < height_threshold:
# #         horizontal_lines.append((x, y, w, h))
# #         # Draw horizontal lines in green
# #         cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # # Sort the horizontal lines based on their vertical position (y-coordinate)
# # horizontal_lines = sorted(horizontal_lines, key=lambda line: line[1])
# # print(horizontal_lines)
# # # Print d
# # # etails of horizontal lines
# # if len(horizontal_lines) >= 3:
# #     print(f"Detected {len(horizontal_lines)} horizontal lines:")
# #     #for idx, (x, y, w, h) in enumerate(horizontal_lines):
# #         #print(f"Line {idx + 1}: Position (x: {x}, y: {y}), Width: {w}, Height: {h}")

# #     # Now extract the label and handwritten parts
# #     first_line_y = horizontal_lines[0][1]  # y-coordinate of the first detected line
# #     second_line_y = horizontal_lines[1][1]  # y-coordinate of the first detected line
# #     third_line_y = horizontal_lines[2][1]  # y-coordinate of the second detected line

# #     # Label part (above the first horizontal line)
# #     label_part = image[first_line_y:second_line_y, :]

# #     # Handwritten part (between the first and second horizontal lines)
# #     handwritten_part = image[second_line_y:third_line_y, :]
# #     # print(handwritten_part)
# #     cv2.imwrite("label_part_2.png", handwritten_part)
# #     # plt.imshow(handwritten_part, cmap='gray')
# #     # plt.savefig("handwritten_part.png")
# # cv2.imwrite("edges_2.png", contour_image)

# # # plt.imshow(contour_image, cmap='gray')
# # # plt.savefig("edges.png")
# # print("process under if")
    
# # print("process complete.")
    
# # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
# # remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
# # cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # for c in cnts:
# #     cv2.drawContours(result, [c], -1, (255,255,255), 5)

# import cv2
# import matplotlib.pyplot as plt

# image = cv2.imread('images/Screenshot 2025-05-15 at 6.13.52 PM.png')
# result = image.copy()
# edges = cv2.Canny(result, 50, 150)
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Remove horizontal lines
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
# remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
# cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(result, [c], -1, (255,255,255), 5)

# horizontal = cv2.erode(result, horizontal_kernel)
# horizontal = cv2.dilate(horizontal, horizontal_kernel)
# cv2.imshow('horizontal', horizontal)
# cv2.waitKey()
# # # Remove vertical lines
# # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
# # remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
# # cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # for c in cnts:
# #     cv2.drawContours(result, [c], -1, (255,255,255), 5)

# cv2.imshow('thresh', thresh)
# cv2.waitKey()
# cv2.imshow('result', result)
# cv2.waitKey()
# cv2.imwrite('result.png', result)
# plt.savefig("line.png")
# cv2.waitKey()


import cv2
import numpy as np

img = cv2.imread('images/Screenshot 2025-05-15 at 6.13.52 PM.png')

if len(img.shape) != 2:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img

gray = cv2.bitwise_not(gray)
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
cv2.THRESH_BINARY, 15, -2)

horizontal = np.copy(bw)

cols = horizontal.shape[1]
horizontal_size = cols // 30

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

img2_dst = cv2.inpaint(img, horizontal, 3, cv2.INPAINT_TELEA)
cv2.imwrite("img2_dst.png", img2_dst)
cv2.imshow('img2_dst', img2_dst)

cv2.imwrite("horizontal_lines_extracted.png", horizontal)
cv2.imshow("horizontal", horizontal)
