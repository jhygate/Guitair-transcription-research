"""
Hough Lines Experiment - Tracks and draws all Hough lines detected on Canny edges.

This experiment uses OpenCV's standard HoughLines function to detect lines in the
Canny edge detector output and visualizes all detected lines.
"""

import cv2
import numpy as np
import time

# ============= PARAMETERS TO TUNE =============
# Canny edge detection
CANNY_LOW_THRESHOLD = 50       # Lower = more edge pixels detected
CANNY_HIGH_THRESHOLD = 150     # Higher = only strong edges kept

# Preprocessing
BLUR_KERNEL_SIZE = 5          # Gaussian blur kernel size (must be odd)

# Hough Line Detection
HOUGH_RHO = 1                 # Distance resolution in pixels
HOUGH_THETA = np.pi / 180     # Angle resolution in radians (1 degree)
HOUGH_THRESHOLD = 100         # Minimum votes to detect a line
HOUGH_MIN_LINE_LENGTH = 50    # Minimum line length (for HoughLinesP)
HOUGH_MAX_LINE_GAP = 10       # Maximum gap between line segments (for HoughLinesP)

# Visualization
LINE_COLOR = (0, 255, 0)      # Green color for detected lines (BGR)
LINE_THICKNESS = 2            # Line thickness in pixels
EDGE_COLOR = (255, 255, 255)  # White color for edge2025-12-14-17-58-26.pngs
# ============================================


def get_canny_edges(frame):
    """
    Extract Canny edge detection from frame.

    Args:
        frame: Input BGR image

    Returns:
        edges: Binary edge image
        edges_bgr: BGR version for visualization
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    # Convert edges to BGR for visualization
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges, edges_bgr


def detect_hough_lines(edges):
    """
    Detect lines using standard Hough transform.

    Args:
        edges: Binary edge image from Canny

    Returns:
        lines: Array of lines in (rho, theta) format, or None if no lines detected
    """
    # HoughLines returns lines in (rho, theta) format
    # rho: distance from origin to the line
    # theta: angle of the line in radians
    lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD)

    return lines


def draw_hough_lines(image, lines):
    """
    Draw all detected Hough lines on the image.

    Args:
        image: BGR image to draw on
        lines: Array of lines in (rho, theta) format from HoughLines

    Returns:
        output: Image with lines drawn
    """
    output = image.copy()

    if lines is None:
        return output

    height, width = image.shape[:2]

    for line in lines:
        rho, theta = line[0]

        # Convert from polar (rho, theta) to Cartesian coordinates
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Find a point on the line
        x0 = cos_theta * rho
        y0 = sin_theta * rho

        # Calculate two points far enough to cross the entire image
        # Extend the line by a large factor
        length = max(width, height) * 2

        x1 = int(x0 + length * (-sin_theta))
        y1 = int(y0 + length * cos_theta)
        x2 = int(x0 - length * (-sin_theta))
        y2 = int(y0 - length * cos_theta) 

        origin = [120, 255/2, 255/2]
        line_colour = tuple([origin[0], origin[1] +  float(cos_theta*255/2), float(origin[2] + sin_theta*255/2)])

        # Draw the line
        cv2.line(output, (x1, y1), (x2, y2), line_colour, LINE_THICKNESS)

    return output


def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("=" * 60)
    print("HOUGH LINES EXPERIMENT")
    print("=" * 60)
    print("\nThis experiment detects and draws all Hough lines on Canny edges")
    print("\nPress 'q' to quit\n")

    print("Parameters:")
    print(f"  Canny thresholds: {CANNY_LOW_THRESHOLD}, {CANNY_HIGH_THRESHOLD}")
    print(f"  Hough rho: {HOUGH_RHO} pixels")
    print(f"  Hough theta: {np.rad2deg(HOUGH_THETA):.1f} degrees")
    print(f"  Hough threshold: {HOUGH_THRESHOLD} votes")
    print("=" * 60)

    # For FPS calculation
    prev_time = time.time()

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Get Canny edges
        edges, edges_bgr = get_canny_edges(frame)

        # Detect Hough lines
        lines = detect_hough_lines(edges)

        # Draw lines on the edge image
        output = draw_hough_lines(edges_bgr, lines)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Count detected lines
        num_lines = len(lines) if lines is not None else 0

        # Display info on frame
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(output, f"Lines detected: {num_lines}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show frames
        cv2.imshow('Canny Edges', edges)
        cv2.imshow('Hough Lines on Edges', output)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nExperiment stopped.")


if __name__ == "__main__":
    main()
