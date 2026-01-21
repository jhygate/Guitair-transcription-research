#!/usr/bin/env python3
"""
Bare bones live guitar note detection using microphone input.
Displays the best guess for the currently played note in the terminal.
"""

import numpy as np
import pyaudio
import sys

# Audio configuration
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
CHANNELS = 1

# Detection thresholds
AMPLITUDE_THRESHOLD = 0.01  # Minimum amplitude to consider a note is being played
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display a note

# Standard guitar tuning lowest note is E2 (82.4 Hz), highest is around E6 (1318 Hz)
MIN_FREQ = 70
MAX_FREQ = 1400

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def frequency_to_note(freq):
    """Convert frequency in Hz to note name and octave."""
    if freq <= 0:
        return None, None

    # A4 = 440 Hz is our reference
    # Formula: n = 12 * log2(f / 440) + 49 (where 49 is A4 on piano)
    note_number = 12 * np.log2(freq / 440.0) + 49
    note_number = int(round(note_number))

    octave = (note_number - 1) // 12
    note_index = (note_number - 1) % 12
    note_name = NOTE_NAMES[note_index]

    return f"{note_name}{octave}", freq


def autocorrelation_detect_pitch(audio_buffer, sample_rate):
    """
    Detect pitch using autocorrelation method.
    Good for guitar because it finds the fundamental frequency reliably.
    """
    # Normalize the audio buffer
    audio_buffer = audio_buffer - np.mean(audio_buffer)

    # Calculate autocorrelation
    autocorr = np.correlate(audio_buffer, audio_buffer, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Find the first peak after the zero lag
    # We look in the range that corresponds to our frequency range
    min_period = int(sample_rate / MAX_FREQ)
    max_period = int(sample_rate / MIN_FREQ)

    if max_period >= len(autocorr):
        return None, 0

    # Find peaks in the autocorrelation
    autocorr_slice = autocorr[min_period:max_period]

    if len(autocorr_slice) == 0:
        return None, 0

    peak_index = np.argmax(autocorr_slice)
    peak_value = autocorr_slice[peak_index]

    # Calculate confidence based on peak prominence
    confidence = peak_value / autocorr[0] if autocorr[0] > 0 else 0

    # Convert period to frequency
    period = peak_index + min_period
    frequency = sample_rate / period

    return frequency, confidence


def audio_callback(in_data, frame_count, time_info, status):
    """Process each audio buffer from the microphone."""
    # Convert bytes to numpy array
    audio_buffer = np.frombuffer(in_data, dtype=np.float32)

    # Check if there's any significant audio signal
    amplitude = np.max(np.abs(audio_buffer))

    if amplitude < AMPLITUDE_THRESHOLD:
        print("\r[No note detected - signal too quiet]" + " " * 20, end='', flush=True)
        return (in_data, pyaudio.paContinue)

    # Detect pitch
    freq, confidence = autocorrelation_detect_pitch(audio_buffer, SAMPLE_RATE)

    if freq is None or confidence < CONFIDENCE_THRESHOLD:
        print("\r[No clear note detected]" + " " * 30, end='', flush=True)
        return (in_data, pyaudio.paContinue)

    # Convert to note name
    note, detected_freq = frequency_to_note(freq)

    if note:
        # Display the result
        confidence_pct = int(confidence * 100)
        print(f"\r{note} ({detected_freq:.1f} Hz) - Confidence: {confidence:.2f}" + " " * 10, end='', flush=True)

    return (in_data, pyaudio.paContinue)


def main():
    """Main function to start live note detection."""
    print("=" * 60)
    print("Live Guitar Note Detection")
    print("=" * 60)
    print("Starting audio capture... Play your guitar!")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    try:
        # Open audio stream
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=BUFFER_SIZE,
            stream_callback=audio_callback
        )

        # Start the stream
        stream.start_stream()

        # Keep the program running
        while stream.is_active():
            try:
                import time
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nStopping...")
                break

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nMake sure your microphone is connected and accessible.", file=sys.stderr)

    finally:
        audio.terminate()
        print("Goodbye!")


if __name__ == "__main__":
    main()
