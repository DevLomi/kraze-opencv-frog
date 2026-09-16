import os
import cv2
import numpy as np


def main():
    # ---------------------------------------------------------
    # Configuration & Setup
    # ---------------------------------------------------------
    input_path = "input_image.jpg"  # Place your test image in the same directory
    output_dir = "output_stages"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the original image
    original = cv2.imread(input_path)
    if original is None:
        raise FileNotFoundError(
            f"Image not found at '{input_path}'. Please check the path."
        )

    # 2. Display the original image
    cv2.imshow("01 - Original Image", original)
    cv2.imwrite(os.path.join(output_dir, "01_original.png"), original)

    # ---------------------------------------------------------
    # Color Conversions
    # ---------------------------------------------------------
    # 3. Convert to Grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "02_grayscale.png"), gray)

    # 4. Convert to HSV
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(output_dir, "03_hsv.png"), hsv)

    # ---------------------------------------------------------
    # Image Filtering (At least two filters)
    # ---------------------------------------------------------
    # Filter 1: Gaussian Blur (smooths high-frequency noise)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Filter 2: Median Blur (removes salt-and-pepper noise while preserving edges)
    filtered = cv2.medianBlur(blurred, 5)
    cv2.imwrite(os.path.join(output_dir, "04_filtered.png"), filtered)

    # ---------------------------------------------------------
    # Edge Detection
    # ---------------------------------------------------------
    # 5. Canny Edge Detection (adjust thresholds to suit your lighting/objects)
    canny_edges = cv2.Canny(filtered, threshold1=50, threshold2=150)
    cv2.imwrite(os.path.join(output_dir, "05_canny_edges.png"), canny_edges)

    # Optional: Morphological close to bridge broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # Contours
    # ---------------------------------------------------------
    # 6. Find and draw contours
    contours, hierarchy = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_canvas = original.copy()
    cv2.drawContours(contour_canvas, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "06_contours.png"), contour_canvas)

    # ---------------------------------------------------------
    # Geometric Detection & Final Results
    # ---------------------------------------------------------
    # 7. Geometric detections: Bounding Box, Min-Area Rect, Enclosing Circle
    final_output = original.copy()
    min_area_threshold = 200  # Filter out tiny noise contours

    detected_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_threshold:
            continue

        detected_count += 1

        # Geometric 1: Upright Bounding Box (Blue)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(final_output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Geometric 2: Minimum-Area Rotated Rectangle (Red)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(final_output, [box], 0, (0, 0, 255), 2)

        # Geometric 3: Minimum Enclosing Circle (Yellow)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(
            final_output, (int(cx), int(cy)), int(radius), (0, 255, 255), 2
        )

        # Geometric 4: Polygon Approximation / Convex Hull (Optional showcase)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(final_output, [hull], -1, (255, 128, 0), 1)

    # Annotate summary on the final image
    cv2.putText(
        final_output,
        f"Detected Items: {detected_count}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        final_output,
        f"Detected Items: {detected_count}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Save and display final result
    cv2.imwrite(os.path.join(output_dir, "07_final_result.png"), final_output)
    cv2.imshow("07 - Final Detection", final_output)

    print(f"Processing complete. {detected_count} objects analyzed.")
    print(f"All stage outputs saved to '{output_dir}/'")

    # Keep windows open until user presses any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()