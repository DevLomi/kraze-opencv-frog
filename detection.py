import os
import cv2
import numpy as np


def get_roi_mask(image):
    """Creates a triangular/trapezoidal Region of Interest mask focused on the road lane."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Focus on the bottom half of the image where lanes sit
    polygon = np.array(
        [
            [
                (int(width * 0.05), height),  # Bottom-left
                (int(width * 0.45), int(height * 0.60)),  # Top-left apex
                (int(width * 0.55), int(height * 0.60)),  # Top-right apex
                (int(width * 0.95), height),  # Bottom-right
            ]
        ],
        np.int32,
    )

    cv2.fillPoly(mask, polygon, 255)
    return mask


def main():
    # ---------------------------------------------------------
    # Setup & Load Image
    # ---------------------------------------------------------
    input_path = "road_input.jpg"  # Replace with your road/dashcam image
    output_dir = "road_output_stages"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load original image
    original = cv2.imread(input_path)
    if original is None:
        raise FileNotFoundError(
            f"Could not load image at '{input_path}'. Check your file name."
        )

    # 2. Display and save original
    cv2.imshow("01 - Original Road", original)
    cv2.imwrite(os.path.join(output_dir, "01_original.png"), original)

    # ---------------------------------------------------------
    # Color Conversions
    # ---------------------------------------------------------
    # 3. Grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "02_grayscale.png"), gray)

    # 4. HSV (Useful for isolating yellow & white lane markings)
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(output_dir, "03_hsv.png"), hsv)

    # ---------------------------------------------------------
    # Filtering (2 distinct filters)
    # ---------------------------------------------------------
    # Filter 1: Bilateral Filter (smoothes road asphalt texture while keeping line edges sharp)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Filter 2: Gaussian Blur (removes any remaining high-frequency pixel noise)
    filtered = cv2.GaussianBlur(bilateral, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "04_filtered.png"), filtered)

    # ---------------------------------------------------------
    # Canny Edge Detection & ROI Masking
    # ---------------------------------------------------------
    # 5. Canny Edge Detection
    edges = cv2.Canny(filtered, threshold1=50, threshold2=150)

    # Restrict edges to the road lane area (cuts out trees, sky, hood of car)
    roi_mask = get_roi_mask(edges)
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    cv2.imwrite(os.path.join(output_dir, "05_canny_edges.png"), masked_edges)

    # ---------------------------------------------------------
    # Contours Detection
    # ---------------------------------------------------------
    # 6. Detect and draw contours of the lane segments
    contours, _ = cv2.findContours(
        masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_canvas = original.copy()
    cv2.drawContours(contour_canvas, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "06_contours.png"), contour_canvas)

    # ---------------------------------------------------------
    # Geometric Detection: Hough Line Detection
    # ---------------------------------------------------------
    # 7. Detect line vectors using Probabilistic Hough Transform
    final_output = original.copy()
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=20,
    )

    line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Optional filter: Ignore near-horizontal lines (e.g. crosswalks/shadows)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15 or abs(angle) > 165:
                continue

            line_count += 1
            # Draw detected road line in bold red
            cv2.line(final_output, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Overlay text feedback
    cv2.putText(
        final_output,
        f"Lanes Segments Detected: {line_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Save and display final result
    cv2.imwrite(os.path.join(output_dir, "07_final_result.png"), final_output)
    cv2.imshow("07 - Final Lane Detection", final_output)

    print(f"Done! Detected {line_count} lane segments.")
    print(f"All project stage images exported to '{output_dir}/'")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()