import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile 
from scipy.fftpack import fft, ifft

fs_5cm, signal_5cm = wavfile.read("original_speech_5cm.wav")

signal_5cm = signal_5cm / np.iinfo(signal_5cm.dtype).max

def set_band_gain(X, f1, f2, gain, fs):
    N = len(X)
    k1 = int(f1 * N / fs)
    k2 = int(f2 * N / fs)
    X[k1:k2+1] = X[k1:k2+1] * gain # positive frequencies
    X[N-k2:N-k1+1] = X[N-k2:N-k1+1] * gain # negative frequencies

def highpass(x, fc, fs):
    X = fft(x)
    set_band_gain(X, 0, fc, 0, fs) # set to 0 so removes those frequencies
    return np.real(ifft(X))

def lowpass(x, fc, fs):
    X = fft(x)
    set_band_gain(X, fc, fs/2, 0, fs)
    return np.real(ifft(X))

X_5cm = fft(signal_5cm)
set_band_gain(X_5cm, 90, 140, 0.5, fs_5cm) # attenuate plosives

clean_5cm = np.real(ifft(X_5cm))

# aural exciter
ae_cutoff = 3000
distortion_gain = 20
ae_mix = 0.5

high_band = highpass(clean_5cm, ae_cutoff, fs_5cm) # cutoff low freq
distorted_high = np.tanh(distortion_gain*high_band)/distortion_gain # add harmonics

# big bottom
bass_cutoff = 190
compression_gain = 5
bass_mix = 0.3

low_band = lowpass(clean_5cm, bass_cutoff, fs_5cm)
compressed_gain = np.tanh(compressed_gain*low_band)/compression_gain

excited_5cm = clean_5cm + (exciter_mix * distorted_high) + (bass_mix * compressed_bass) # mix back into input signal

wavfile.write("exciter_speech.wav", fs_5cm, excited_5cm.astype(np.float32))