import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

fs = 44100
duration = 5

recording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
sd.wait()

name = f"recording_{timestamp}.wav"

write(name, fs, recording)

print(f"Saved {name}")
