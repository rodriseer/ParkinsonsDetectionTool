import sounddevice as sd
from scipy.io.wavfile import write
import wavio


"""
using sounddevice to capture a 5 seconds voice from user
saves as voiceOutput.wav file

change parameters as needed, such as 'duration = int'

will implement the voice data extraction through this file
"""
def record_audio(filename, duration=5, fs=44100):
    print(f"Say 'Ah' for 15 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("Recording complete!")
    wavio.write(filename, recording, fs, sampwidth=2)

record_audio('voiceOutput.wav', duration=10)
