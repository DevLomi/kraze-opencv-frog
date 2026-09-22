import cv2
import os
import numpy as np

IMAGE_PATH = "road_in_norway.jpg"


# 1. Load an image using OpenCV
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]
output_dir = "road_output_stages"
os.makedirs(output_dir, exist_ok=True)

# 2. Save original image 
cv2.imwrite(os.path.join(output_dir, "01_original.png"), img)


# 3. grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, "02_grayscale.png"), gray)

# 4. Convert the image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite(os.path.join(output_dir, "03_hsv.png"), hsv)

# 5. two image filters
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
filtered = cv2.medianBlur(gaussian, 5)
cv2.imwrite(os.path.join(output_dir, "04_filtered.png"), filtered)


# 6.Canny edge detection (focused on road region)

edges = cv2.Canny(filtered, 50, 150)
# Exclude sky and mountains above the road horizon
edges[:int(h * 0.35), :] = 0
cv2.imwrite(os.path.join(output_dir, "05_canny_edges.png"), edges)


# 7. Detect and draw contours using road colors (gray/black asphalt)
    # low saturation gray/black across dark to sunlit brightness
road_mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 90, 240]))
road_mask[:int(h * 0.35), :] = 0  # Ignore sky and mountains

contours, hierarchy = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir, "06_contours.png"), contour_img)


# 8. geometric detection
final_output = img.copy()

# Geometric Detection 1: Multiple bounding boxes narrowing down the road
num_bands = 5
band_h = int((h - int(h * 0.35)) / num_bands)
for i in range(num_bands):
    y1 = int(h * 0.35) + i * band_h
    y2 = y1 + band_h if i < num_bands - 1 else h
    strip = road_mask[y1:y2, :].copy()
    cnts, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            bx, by, bw, bh = cv2.boundingRect(c)
            abs_y = y1 + by
            cv2.rectangle(final_output, (bx, abs_y), (bx + bw, abs_y + bh), (0, 255, 255), 2)

# Geometric Detection 2: Hough line detection for road borders and railings
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=15)
if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        # Ignore flat horizontal noise lines
        if abs(y2 - y1) > 15:
            xm, ym = (x1 + x2) // 2, (y1 + y2) // 2
            # Interpolate road and railing corridor from horizon to bottom
            t = (ym - h * 0.35) / (h * 0.65)
            min_x = (1.0 - t) * (w * 0.35)
            max_x = w * 0.75 + t * (w * 0.25)
            # Keep lines located on the road borders and railings
            if min_x <= xm <= max_x:
                cv2.line(final_output, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 9. final process output
scale = min(1280 / w, 720 / h)
win_w, win_h = int(w * scale), int(h * scale)
cv2.namedWindow("Road Detection - Final Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Road Detection - Final Result", win_w, win_h)
cv2.imshow("Road Detection - Final Result", final_output)




# 10. Save processed output
cv2.imwrite(os.path.join(output_dir, "07_final_result.png"), final_output)
print(f"Done! Output saved to '{output_dir}/07_final_result.png'")
cv2.waitKey(0)
cv2.destroyAllWindows()

