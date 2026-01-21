# Live Guitar Note Detector - Setup Instructions

## Installation

### 1. macOS: Install PortAudio (required for PyAudio)

```bash
brew install portaudio
```

### 2. Setup dependencies with uv

```bash
# Sync dependencies from pyproject.toml
uv sync
```

## Running the Detector

```bash
uv run live_note_detector.py
```

## Usage

1. Start the program
2. Play single notes on your guitar
3. The terminal will display the detected note in real-time
4. Press `Ctrl+C` to stop

## Expected Output

```
E2 (82.4 Hz) - Confidence: 0.89
A2 (110.1 Hz) - Confidence: 0.85
[No note detected - signal too quiet]
D3 (146.8 Hz) - Confidence: 0.92
```

## Troubleshooting

- **"No note detected - signal too quiet"**: Play louder or move closer to the microphone
- **Microphone permission error**: Grant microphone access to Terminal in System Settings
- **Import errors**: Make sure numpy and pyaudio are installed in your Python environment
- **Inaccurate detection**: Try adjusting thresholds in the script (AMPLITUDE_THRESHOLD, CONFIDENCE_THRESHOLD)

## Adjustable Parameters

In the script, you can modify:

- `BUFFER_SIZE`: Larger = more accurate but higher latency (try 4096)
- `AMPLITUDE_THRESHOLD`: Lower = more sensitive (try 0.005)
- `CONFIDENCE_THRESHOLD`: Lower = show more uncertain detections (try 0.2)
