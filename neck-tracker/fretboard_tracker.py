"""
Fretboard tracker using custom parallel Hough transform.

Uses accumulator voting in (d, θ) parameter space to detect parallel lines.
For each Canny edge point (x, y), we vote for line families defined by:
    x*cos(θ) + y*sin(θ) = k*d

where k ∈ {0, 1, 2, ..., K_MAX} represents parallel lines with spacing d.
"""

import cv2
import numpy as np
import time

# ============= PARAMETERS TO TUNE =============
# Canny edge detection
CANNY_LOW_THRESHOLD = 1       # Lower = more edge pixels detected
CANNY_HIGH_THRESHOLD = 60     # Higher = only strong edges kept

# Preprocessing
BLUR_KERNEL_SIZE = 5          # Gaussian blur kernel size (must be odd)

# ============= PARALLEL HOUGH PARAMETERS =============
# Number of parallel lines to search for (k = 0, 1, 2, ..., K_MAX)
K_MAX = 10              # For guitar frets (typically ~12-24 frets visible)

# Distance parameter space
D_MIN = 10                    # Minimum line spacing in pixels
D_MAX = 100                   # Maximum line spacing in pixels
D_RESOLUTION = 10            # Step size for d parameter (pixels)

# Angle parameter space
THETA_MIN = 0                 # Minimum angle in degrees
THETA_MAX = 180               # Maximum angle in degrees
THETA_RESOLUTION = 10          # Step size for θ parameter (degrees)

# Accumulator voting
DISTANCE_THRESHOLD = 5.0      # Maximum distance (pixels) from point to line to vote
ACCUMULATOR_THRESHOLD = 100   # Minimum votes required for a line family to be detected

# Visualization
TOP_N_PERCENT = 1             # Draw top N% of candidate line families (e.g., 5 = top 5%)
SHOW_ALL_HOUGH_LINES = True   # Draw all Hough candidates with votes > MIN_VOTES_TO_SHOW
MIN_VOTES_TO_SHOW = 50        # Minimum votes to show a Hough candidate
LINE_COLOR = (0, 255, 0)      # Green color for detected lines (BGR)
LINE_THICKNESS = 2            # Line thickness in pixels
# ============================================


def get_edge_points(frame):
    """
    Extract Canny edge points from frame.

    Args:
        frame: Input BGR image

    Returns:
        edge_points: Nx2 array of (x, y) coordinates of edge pixels
        edges: Binary edge image for visualization
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    # Get coordinates of edge pixels
    # np.nonzero returns (row_indices, col_indices), we want (x, y)
    y_coords, x_coords = np.nonzero(edges)
    edge_points = np.column_stack((x_coords, y_coords))

    return edge_points, edges


def parallel_hough_transform(edge_points, frame_shape):
    """
    Custom Hough transform for detecting families of parallel lines.
    VECTORIZED VERSION - much faster than nested loops.

    For each edge point (x, y) and parameter pair (d, θ), we check if the point
    is close to ANY of the k parallel lines defined by:
        x*cos(θ) + y*sin(θ) = k*d,  for k = 0, 1, 2, ..., K_MAX

    Args:
        edge_points: Nx2 array of (x, y) edge pixel coordinates
        frame_shape: (height, width) of the image

    Returns:
        best_d: Optimal line spacing parameter (pixels)
        best_theta: Optimal angle parameter (radians)
        best_votes: Number of votes for the best parameter pair
    """
    # Build parameter space
    d_values = np.arange(D_MIN, D_MAX + D_RESOLUTION, D_RESOLUTION)
    theta_values_deg = np.arange(THETA_MIN, THETA_MAX + THETA_RESOLUTION, THETA_RESOLUTION)
    theta_values_rad = np.deg2rad(theta_values_deg)

    # Initialize accumulator: accumulator[d_idx, theta_idx]
    accumulator = np.zeros((len(d_values), len(theta_values_rad)), dtype=np.int32)

    # Extract x and y coordinates
    x_coords = edge_points[:, 0]  # Shape: (N,)
    y_coords = edge_points[:, 1]  # Shape: (N,)

    # For each parameter combination (d, θ)
    for d_idx, d in enumerate(d_values):
        for theta_idx, theta in enumerate(theta_values_rad):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Vectorized: compute rho for ALL edge points at once
            # rho shape: (N,) where N is number of edge points
            rho = x_coords * cos_theta + y_coords * sin_theta

            # Vectorized: find closest k for all points
            if d > 0:
                k_closest = np.round(rho / d)
                k_closest = np.clip(k_closest, 0, K_MAX)  # Keep k in valid range

                # Vectorized: compute distance for all points
                distance = np.abs(rho - k_closest * d)

                # Vectorized: count votes (points within threshold)
                votes = np.sum(distance < DISTANCE_THRESHOLD)

                accumulator[d_idx, theta_idx] = votes

    # Find the parameter pair with maximum votes
    max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    best_d_idx, best_theta_idx = max_idx

    best_d = d_values[best_d_idx]
    best_theta = theta_values_rad[best_theta_idx]
    best_votes = accumulator[best_d_idx, best_theta_idx]

    return best_d, best_theta, best_votes, accumulator


def draw_parallel_lines(frame, d, theta, color=None):
    """
    Draw the family of parallel lines on the frame.

    Lines are defined by: x*cos(θ) + y*sin(θ) = k*d

    Args:
        frame: Input BGR image
        d: Line spacing parameter (pixels)
        theta: Angle parameter (radians)
        color: Optional BGR color tuple (defaults to LINE_COLOR)

    Returns:
        output: Frame with lines drawn
    """
    output = frame.copy()
    height, width = frame.shape[:2]

    if color is None:
        color = LINE_COLOR

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Draw each of the k parallel lines
    for k in range(K_MAX + 1):
        rho = k * d

        # Convert from polar (rho, theta) to Cartesian line equation
        # Line equation: x*cos(θ) + y*sin(θ) = rho

        # Find two points on the line within the frame bounds
        if abs(sin_theta) > 0.001:  # Not horizontal
            # Compute x for y=0 and y=height
            x0 = int(rho / cos_theta) if abs(cos_theta) > 0.001 else 0
            y0 = 0

            x1 = int((rho - height * sin_theta) / cos_theta) if abs(cos_theta) > 0.001 else width
            y1 = height
        else:  # Horizontal line
            x0 = 0
            y0 = int(rho / sin_theta) if abs(sin_theta) > 0.001 else 0

            x1 = width
            y1 = y0

        # Draw the line
        cv2.line(output, (x0, y0), (x1, y1), color, LINE_THICKNESS)

    return output


def get_top_candidates(accumulator, d_values, theta_values_rad, top_n_percent):
    """
    Get the top N% of candidate (d, θ) pairs from the accumulator.

    Args:
        accumulator: 2D array of vote counts
        d_values: Array of d parameter values
        theta_values_rad: Array of θ parameter values (radians)
        top_n_percent: Percentage of top candidates to return (0-100)

    Returns:
        candidates: List of tuples (d, theta, votes) sorted by votes descending
    """
    # Flatten accumulator and get indices
    flat_accumulator = accumulator.flatten()

    # Calculate how many candidates to return
    total_candidates = len(flat_accumulator)
    num_top = max(1, int(total_candidates * top_n_percent / 100))

    # Get indices of top N% candidates
    top_indices = np.argpartition(flat_accumulator, -num_top)[-num_top:]

    # Sort these top indices by vote count (descending)
    top_indices = top_indices[np.argsort(flat_accumulator[top_indices])[::-1]]

    # Convert flat indices back to 2D (d_idx, theta_idx)
    candidates = []
    for flat_idx in top_indices:
        d_idx, theta_idx = np.unravel_index(flat_idx, accumulator.shape)
        d = d_values[d_idx]
        theta = theta_values_rad[theta_idx]
        votes = accumulator[d_idx, theta_idx]
        candidates.append((d, theta, votes))

    return candidates


def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("=" * 60)
    print("PARALLEL HOUGH FRETBOARD TRACKER")
    print("=" * 60)
    print("\nAlgorithm:")
    print("  For each edge point (x,y), vote in (d,θ) parameter space")
    print("  Line family: x*cos(θ) + y*sin(θ) = k*d, k = 0..K_MAX")
    print("  Vote if distance to nearest parallel line < threshold")
    print("\nPress 'q' to quit\n")

    print("Parameters:")
    print(f"  Canny thresholds: {CANNY_LOW_THRESHOLD}, {CANNY_HIGH_THRESHOLD}")
    print(f"  K_MAX (# parallel lines): {K_MAX}")
    print(f"  d range: [{D_MIN}, {D_MAX}] pixels, step={D_RESOLUTION}")
    print(f"  θ range: [{THETA_MIN}, {THETA_MAX}] degrees, step={THETA_RESOLUTION}")
    print(f"  Distance threshold: {DISTANCE_THRESHOLD} pixels")
    print(f"  Accumulator threshold: {ACCUMULATOR_THRESHOLD} votes")
    print("=" * 60)

    # For FPS calculation
    prev_time = time.time()

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame_height, frame_width = frame.shape[:2]

        # Extract edge points from Canny
        edge_points, edges = get_edge_points(frame)

        # Start with original frame
        output = frame.copy()

        # Overlay edges in white
        output[edges > 0] = [255, 255, 255]
        output[edges == 0] = [0, 0, 0]

        # Run parallel Hough transform
        if len(edge_points) > 0:
            # best_d, best_theta, best_votes, accumulator = parallel_hough_transform(
            #     edge_points, (frame_height, frame_width)
            # )

            # Get parameter space arrays for candidate extraction
            d_values = np.arange(D_MIN, D_MAX + D_RESOLUTION, D_RESOLUTION)
            theta_values_deg = np.arange(THETA_MIN, THETA_MAX + THETA_RESOLUTION, THETA_RESOLUTION)
            theta_values_rad = np.deg2rad(theta_values_deg)

            # # Get top N% candidates
            # top_candidates = get_top_candidates(accumulator, d_values, theta_values_rad, TOP_N_PERCENT)

            # # Optionally draw ALL Hough line candidates (dimmed)
            if SHOW_ALL_HOUGH_LINES:
                for d_idx, d in enumerate(d_values):
                    for theta_idx, theta in enumerate(theta_values_rad):
                        votes = accumulator[d_idx, theta_idx]
                        if votes >= MIN_VOTES_TO_SHOW:
                            # Draw dimmed gray lines for all candidates
                            # Brightness proportional to votes
                            max_votes = np.max(accumulator)
                            brightness = int(50 + (votes / max_votes) * 100)  # 50-150 range
                            gray_color = (brightness, brightness, brightness)
                            output = draw_parallel_lines(output, d, theta, color=gray_color)

            # # Draw all top candidates with different bright colors (on top)
            # colors = [
            #     (0, 255, 0),    # Green
            #     (255, 0, 0),    # Blue
            #     (0, 255, 255),  # Yellow
            #     (255, 0, 255),  # Magenta
            #     (0, 165, 255),  # Orange
            # ]

            # for idx, (d, theta, votes) in enumerate(top_candidates):
            #     if votes >= ACCUMULATOR_THRESHOLD:
            #         color = colors[idx % len(colors)]
            #         output = draw_parallel_lines(output, d, theta, color=color)
        else:
            best_d, best_theta, best_votes = 0, 0, 0
            top_candidates = []

        # # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display info on frame
        theta_deg = np.rad2deg(best_theta)
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Edge points: {len(edge_points)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Top {TOP_N_PERCENT}% candidates: {len(top_candidates)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Best d: {best_d:.1f} px", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Best theta: {theta_deg:.1f} deg", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Best votes: {best_votes}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Parallel Hough Transform', output)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nTracker stopped.")


if __name__ == "__main__":
    main()
