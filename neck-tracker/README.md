# Fretboard Tracker - Parallel Hough Transform

Computer vision prototype for detecting families of parallel lines on a guitar fretboard using a custom Hough transform algorithm.

## Setup

This project uses `uv` for dependency management.

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install dependencies

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Usage

```bash
python fretboard_tracker.py
```

Point your webcam at a guitar fretboard. The script will:
- Extract edge points using Canny edge detection
- Search parameter space (d, θ) for families of parallel lines
- Display the best-fitting parallel line family with K_MAX lines
- Show real-time parameters: d (spacing), θ (angle), votes (confidence)

Press `q` to quit.

## Algorithm

The tracker uses a custom **parallel Hough transform** to detect families of evenly-spaced parallel lines (like frets):

### Mathematical Model

For each parameter pair **(d, θ)** where:
- **d** = spacing between parallel lines (pixels)
- **θ** = angle of the line family (radians)

We define a family of parallel lines:

```
x·cos(θ) + y·sin(θ) = k·d,  for k = 0, 1, 2, ..., K_MAX
```

### Voting Algorithm

1. **Extract edge points**: Use Canny to get (x, y) coordinates of edge pixels
2. **For each (d, θ) pair in parameter space**:
   - For each edge point (x, y):
     - Compute ρ = x·cos(θ) + y·sin(θ)
     - Find k_closest = round(ρ / d) that minimizes |ρ - k·d|
     - Calculate distance = |ρ - k_closest·d|
     - **If distance < DISTANCE_THRESHOLD**: vote for this (d, θ)
3. **Return (d, θ) with maximum votes**

### Key Insight

Unlike standard Hough transform (which detects individual lines), this accumulator votes for *entire families* of parallel lines simultaneously. A point votes for (d, θ) if it's close to ANY of the k parallel lines in that family.

## Parameters

All parameters are clearly defined at the top of [fretboard_tracker.py](fretboard_tracker.py). Here are the key ones:

### Edge Detection
```python
CANNY_LOW_THRESHOLD = 1       # Lower = more edge pixels (more sensitive)
CANNY_HIGH_THRESHOLD = 60     # Upper threshold for strong edges
BLUR_KERNEL_SIZE = 5          # Gaussian blur kernel (reduces noise)
```

### Parallel Line Search
```python
K_MAX = 18                    # Number of parallel lines to detect (0..18)
                              # For frets: typically 12-24 visible frets

D_MIN = 10                    # Minimum spacing between lines (pixels)
D_MAX = 100                   # Maximum spacing between lines (pixels)
D_RESOLUTION = 1              # Step size for d parameter

THETA_MIN = 0                 # Minimum angle (degrees)
THETA_MAX = 180               # Maximum angle (degrees)
THETA_RESOLUTION = 1          # Step size for θ parameter
```

### Accumulator Voting
```python
DISTANCE_THRESHOLD = 5.0      # Max distance (px) from point to line to vote
                              # Lower = stricter, fewer votes
                              # Higher = more permissive, more votes

ACCUMULATOR_THRESHOLD = 100   # Minimum votes to display a line family
                              # Lower = detect weaker patterns
                              # Higher = only strong parallel line families
```

### Tuning Tips

**Not detecting frets?**
- Decrease `ACCUMULATOR_THRESHOLD` (try 50)
- Increase `DISTANCE_THRESHOLD` (try 8-10)
- Lower `CANNY_LOW_THRESHOLD` (try 1)
- Adjust `D_MIN` / `D_MAX` to match your fret spacing

**Detecting wrong patterns?**
- Increase `ACCUMULATOR_THRESHOLD`
- Decrease `DISTANCE_THRESHOLD`
- Narrow the `THETA` range if you know the approximate angle

**Performance issues?**
- Increase `D_RESOLUTION` and `THETA_RESOLUTION` (try 2-5)
- Decrease `K_MAX` if you don't need many lines
- Reduce `D_MAX - D_MIN` range

## Variable Definitions

### Input Variables
- `edge_points` — Nx2 numpy array of (x, y) coordinates from Canny edge detection
- `frame_shape` — (height, width) of the input image

### Parameter Space
- `d` — spacing between parallel lines (pixels), scanned from D_MIN to D_MAX
- `θ` (theta) — angle of line family (radians), scanned from THETA_MIN to THETA_MAX
- `k` — line index within family, k ∈ {0, 1, 2, ..., K_MAX}

### Computed Values
- `ρ` (rho) — distance from origin to line: ρ = x·cos(θ) + y·sin(θ)
- `k_closest` — nearest integer k such that k·d ≈ ρ
- `distance` — perpendicular distance from point to the k_closest line: |ρ - k_closest·d|
- `accumulator[d_idx, θ_idx]` — vote count for parameter pair (d, θ)

### Output Variables
- `best_d` — optimal line spacing (pixels)
- `best_theta` — optimal angle (radians)
- `best_votes` — number of edge points that voted for this (d, θ)

## Next Steps

To build a complete fretboard tracker:

1. **Detect both frets AND strings** — Run algorithm twice with different θ ranges
2. **Find fretboard corners** — Intersect outermost fret/string lines
3. **Perspective correction** — Use cv2.getPerspectiveTransform for top-down view
4. **Temporal tracking** — Track parameters across frames with optical flow
5. **Finger detection** — Add MediaPipe Hands and map to fret/string coordinates
