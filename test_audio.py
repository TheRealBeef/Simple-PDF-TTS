import numpy as np
import sounddevice as sd

# Generate a 1-second sine wave at 440 Hz
fs = 44100  # Sample rate
duration = 1.0  # Duration in seconds
frequency = 440.0  # Frequency in Hz

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
x = 0.5 * np.sin(2 * np.pi * frequency * t)

print("Playing sine wave...")
sd.play(x, fs)
sd.wait()
print("Playback finished.")



# Needed to run
# sudo apt-get update
# sudo apt-get install --reinstall alsa-base pulseaudio
# sudo alsa force-reload