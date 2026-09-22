# FroggerCV 🐸

A simple OpenCV road detection program that analyzes road surfaces, boundaries, and railings to map a safe crossing path.

## Requirements

Install the dependencies:

```bash
pip install opencv-python numpy
```

## How to Run

```bash
python detection.py
```

## How It Works

1. **Preprocessing**: Converts the image to grayscale and HSV, then applies Gaussian and Median filters.
2. **Edge & Horizon Filtering**: Detects edges with Canny and ignores the sky/mountains above the road.
3. **Road Contours**: Isolates the asphalt color to find road boundaries.
4. **Geometric Detection**: Draws bounding boxes (yellow) down the road corridor and Hough lines (red) along railings and borders.

All intermediate steps and the final result are saved to the `road_output_stages/` folder.
