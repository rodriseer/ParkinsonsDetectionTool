import sounddevice as sd
from scipy.io.wavfile import write
import wavio
"""
using sounddevice to capture a 5 seconds voice from user.
saves the voice recording as voiceOutput.wav file

change parameters as needed, such as 'duration(int) = value'

futurely -> will implement the voice data extraction through this file
"""
# length of the recording
duration = 5
# frequency, 44100 or 48000, tweak as needed
fs = 48000
# sound capture channel, 1 or 2, tweak as needed
channels = 2

def record_audio(filename, duration, fs):
    print(f"Say 'Ah' for 5 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels, dtype='float64')
    sd.wait()
    print("Recording complete!")
    wavio.write(filename, recording, fs, sampwidth=2)

record_audio('voiceOutput.wav', duration=10)
