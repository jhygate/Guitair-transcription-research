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
EDGE_COLOR = (255, 255, 255)  # White color for edges.              
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
    Detect lines using standard Hough transform and find contributing edge pixels.

    Args:
        edges: Binary edge image from Canny

    Returns:
        lines_with_segments: List of tuples (rho, theta, start_point, end_point)
                            where start_point and end_point are the endpoints of the
                            longest continuous segment
    """
    # HoughLines returns lines in (rho, theta) format
    # rho: distance from origin to the line
    # theta: angle of the line in radians
    lines = cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD)

    if lines is None:
        return None

    # Dilate edges to connect nearby pixels
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Get all edge pixel coordinates (y, x format)
    edge_points = np.argwhere(dilated_edges > 0)

    # Convert to (x, y) format for easier calculation
    edge_x = edge_points[:, 1]
    edge_y = edge_points[:, 0]

    lines_with_segments = []

    for line in lines:
        rho, theta = line[0]

        # Vectorized distance calculation
        # A pixel (x, y) contributes if: x*cos(theta) + y*sin(theta) ≈ rho
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Calculate distances for all edge pixels at once
        distances = np.abs(edge_x * cos_theta + edge_y * sin_theta - rho)

        # Find pixels within tolerance
        mask = distances < 1.5

        if np.sum(mask) >= 2:
            # Get contributing pixels
            contributing_x = edge_x[mask]
            contributing_y = edge_y[mask]
            contributing_pixels = np.column_stack((contributing_x, contributing_y))

            # Project all points onto the line direction
            line_direction = np.array([-sin_theta, cos_theta])
            projections = contributing_pixels @ line_direction

            # Sort pixels by their projection along the line
            sorted_indices = np.argsort(projections)
            sorted_projections = projections[sorted_indices]
            sorted_pixels = contributing_pixels[sorted_indices]

            # Find the longest continuous segment
            # Pixels are continuous if their projection difference is small
            max_gap = 3.0  # Maximum gap in projection space to be considered continuous

            best_start_idx = 0
            best_end_idx = 0
            best_length = 0

            current_start_idx = 0

            for i in range(1, len(sorted_projections)):
                gap = sorted_projections[i] - sorted_projections[i-1]

                if gap > max_gap:
                    # Found a gap, check if current segment is the longest
                    current_length = i - current_start_idx
                    if current_length > best_length:
                        best_length = current_length
                        best_start_idx = current_start_idx
                        best_end_idx = i - 1

                    # Start new segment
                    current_start_idx = i

            # Check the final segment
            current_length = len(sorted_projections) - current_start_idx
            if current_length > best_length:
                best_length = current_length
                best_start_idx = current_start_idx
                best_end_idx = len(sorted_projections) - 1

            # Only add if we have at least 2 pixels in the longest segment
            if best_length >= 2:
                start_point = tuple(sorted_pixels[best_start_idx].astype(int))
                end_point = tuple(sorted_pixels[best_end_idx].astype(int))

                lines_with_segments.append((rho, theta, start_point, end_point))

    return lines_with_segments if lines_with_segments else None


def draw_hough_lines(image, lines):
    """
    Draw line segments between actual start and end points of contributing pixels.

    Args:
        image: BGR image to draw on
        lines: List of tuples (rho, theta, start_point, end_point)

    Returns:
        output: Image with line segments drawn
    """
    output = image.copy()

    if lines is None:
        return output

    for line_data in lines:
        _, theta, start_point, end_point = line_data

        # Color based on line angle for visualization
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        origin = [120, 255/2, 255/2]
        line_colour = tuple([origin[0], origin[1] + float(cos_theta*255/2), float(origin[2] + sin_theta*255/2)])

        # Draw only the segment between start and end points
        cv2.line(output, start_point, end_point, line_colour, LINE_THICKNESS)

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
