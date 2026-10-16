import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile 
from scipy.fftpack import fft, ifft

fs_5cm, signal_5cm = wavfile.read("original_speech_5cm.wav")

signal_5cm = signal_5cm / np.iinfo(signal_5cm.dtype).max

def set_band_gain(x, f1, f2, gain_db, fs):
    gain = 10**(gain_db / 20)
    N = len(x)
    k1 = int(f1 * N / fs)
    k2 = int(f2 * N / fs)
    x[k1:k2+1] = x[k1:k2+1] * gain # positive frequencies
    x[N-k2:N-k1+1] = x[N-k2:N-k1+1] * gain # negative frequencies

# windowing bandpass, to prevent ringing
def set_band_gain_window(x, f1, f2, outer_gain_db, transition_width, fs):
    outer_gain = 10**(outer_gain_db / 20)
    N = len(x)
    bin_freq = np.arange(N)*(fs/N)
    bin_freq = np.minimum(bin_freq, fs - bin_freq)
    hz_outside_band = np.maximum(f1-bin_freq, bin_freq-f2)
    ramp_position = np.clip(hz_outside_band/transition_width,0,1)
    bin_gain = 1 + (outer_gain-1)*ramp_position
    x[:]=x*bin_gain

def highpass(x, fc, fs):
    X = fft(x)
    set_band_gain(X, 0, fc, -np.inf, fs) # set to -inf to remove those freqs
    return np.real(ifft(X))

def lowpass(x, fc, fs):
    X = fft(x)
    set_band_gain(X, fc, fs/2, -np.inf, fs)
    return np.real(ifft(X))

X_5cm = fft(signal_5cm)
set_band_gain(x=X_5cm, f1=90, f2=140, gain_db=-6, fs=fs_5cm) # attenuate plosives

clean_5cm = np.real(ifft(X_5cm))

# aural exciter
ae_cutoff = 3000
distortion_gain = 20
ae_mix = 0.5

high_band = highpass(clean_5cm, ae_cutoff, fs_5cm) # cutoff low freq
harmonics = np.tanh(distortion_gain*high_band)/distortion_gain # add harmonics

# big bottom
bass_cutoff = 190
compression_gain = 5
bass_mix = 0.3

low_band = lowpass(clean_5cm, bass_cutoff, fs_5cm)
soft_clipping = np.tanh(compression_gain *low_band)/compression_gain

# select output harmonics
harm_band_low = 4000
harm_band_high = 8000
transition_width = 500

X_harm = fft(harmonics)

set_band_gain_window(X_harm, harm_band_low, harm_band_high, -np.inf, transition_width, fs_5cm)
output_harmonics = np.real(ifft(X_harm))

excited_5cm = clean_5cm + (ae_mix * output_harmonics) + (bass_mix * soft_clipping) # mix back into input signal

wavfile.write("exciter_speech.wav", fs_5cm, excited_5cm.astype(np.float32))