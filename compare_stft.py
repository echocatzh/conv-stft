import torch
from conv_stft import STFT
import numpy as np
import librosa 
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

audio = librosa.load(librosa.util.example_audio_file())[0]
device = 'cpu'
win_len = 1024
win_hop = 256
fft_len = 1024
window = 'hann'
librosa_stft = librosa.stft(audio, n_fft=fft_len, hop_length=win_hop, window=window)
_magnitude = np.abs(librosa_stft)

audio = torch.FloatTensor(audio)
audio = audio.unsqueeze(0)
audio = audio.to(device)

stft = STFT(
    fft_len=fft_len, 
    win_hop=win_hop, 
    win_len=win_len,
    win_type=window,
).to(device)

magnitude, phase = stft.transform(audio)
plt.figure(figsize=(6, 6))
plt.subplot(211)
plt.title('PyTorch STFT magnitude')
plt.xlabel('Frames')
plt.ylabel('FFT bin')
plt.imshow(20*np.log10(1+magnitude[0].cpu().data.numpy()), aspect='auto', origin='lower')

plt.subplot(212)
plt.title('Librosa STFT magnitude')
plt.xlabel('Frames')
plt.ylabel('FFT bin')
plt.imshow(20*np.log10(1+_magnitude), aspect='auto', origin='lower')
plt.tight_layout()
plt.savefig('images/stft.png')

output = stft.inverse(magnitude, phase)
output = output.cpu().data.numpy()[..., :]
audio = audio.cpu().data.numpy()[..., :]
print(np.mean((output - audio) ** 2)) # on order of 1e-15