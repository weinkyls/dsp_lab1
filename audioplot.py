import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile 
from scipy.fftpack import fft, ifft

#EXCERCISE 1
#import sampling rate and signal in np array
fs_1m, signal_1m = wavfile.read("original_speech_1m.wav")
fs_5cm, signal_5cm = wavfile.read("original_speech_5cm.wav")


# 1i) normalise
signal_1m = signal_1m / np.iinfo(signal_1m.dtype).max
signal_5cm = signal_5cm / np.iinfo(signal_5cm.dtype).max

# construct time array
N_1m = signal_1m.size 
time_1m = np.arange(N_1m)/fs_1m
N_5cm = signal_5cm.size 
time_5cm = np.arange(N_5cm)/fs_5cm

# plot in time domain, normalised amplitudes v time
plt.figure()
plt.subplot(2,2,1)
plt.plot(time_1m, signal_1m)
plt.xlabel("Time (s)")
plt.ylabel("Normalised Amplitude")
plt.grid(True)
plt.title("Time Domain Plot: 1 m Signal")

plt.subplot(2,2,2)
plt.plot(time_5cm, signal_5cm)
plt.xlabel("Time (s)")
plt.ylabel("Normalised Amplitude")
plt.grid(True)
plt.title("Time Domain Plot: 5 cm Signal")

# 1ii) fft
fft_5cm = fft(signal_5cm)
fft_1m = fft(signal_1m)

# construct freq array (positive only)
freq_1m = np.arange(N_1m//2) * fs_1m / N_1m
freq_5cm = np.arange(N_5cm//2) * fs_5cm / N_5cm

# log + magnitude
amp_5cm = 20*np.log10(np.abs(fft_5cm))
amp_1m = 20*np.log10(np.abs(fft_1m))
peaks_1m = np.array([186.1, 204.6, 296.17, 350])
peaks_5cm = np.array([166.95, 202.92, 267.29])

plt.subplot(2,2,3)
plt.plot(freq_1m, amp_1m[:N_1m//2])
plt.xscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.grid(True, which="both", linestyle=":", linewidth=0.5)
for f0 in peaks_1m:
    plt.axvline(f0, color="r", linestyle="--", linewidth=0.8, label="Fundamental Frequency" if f0 == peaks_1m[0] else None)
plt.axvspan(820, 10960, color="g", alpha=0.2, label="Consonant") # consonant range
plt.axvspan(10960, fs_5cm/2, color="grey", alpha=0.2, label="Noise")  # noise region
plt.axvspan(20, 94.5, color="grey", alpha=0.2)
plt.legend()
plt.title("1 m fft")

plt.subplot(2,2,4)
plt.plot(freq_5cm, amp_5cm[:N_5cm//2])
plt.xscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.grid(True, which="both", linestyle=":", linewidth=0.5)
for f0 in peaks_5cm:
    plt.axvline(f0, color="r", linestyle="--", linewidth=0.8, label="Fundamental Frequency" if f0 == peaks_5cm[0] else None)
plt.axvspan(1176, 10720, color="g", alpha=0.2, label="Consonant") # consonant range
plt.axvspan(10720, fs_5cm/2, color="grey", alpha=0.2, label="Noise")  # noise region
plt.axvspan(20, 90, color="grey", alpha=0.2)
plt.legend()
plt.title("5 cm fft")
plt.show()

# EXCERCISE 2

# windowing bandpass, to prevent ringing
def set_band_gain_window(x, f1, f2, inner_gain_db, outer_gain_db, transition_width, fs):
    outer_gain = 10**(outer_gain_db / 20)
    inner_gain = 10**(inner_gain_db / 20)
    N = len(x)
    bin_freq = np.arange(N)*(fs/N)
    bin_freq = np.minimum(bin_freq, fs - bin_freq)
    hz_outside_band = np.maximum(f1-bin_freq, bin_freq-f2)
    ramp_position = np.clip(hz_outside_band/transition_width,0,1)
    bin_gain = inner_gain + (outer_gain-inner_gain)*ramp_position
    x[:]=x*bin_gain

X_5cm = fft(signal_5cm)

set_band_gain_window(X_5cm, f1=90, f2=10720, inner_gain_db=0, outer_gain_db=-np.inf, transition_width=50, fs=fs_5cm) # f1 and f2 from task 1
X_5cm_denoised = X_5cm.copy()
set_band_gain_window(x=X_5cm, f1=2000, f2=6000, inner_gain_db=3.5, outer_gain_db=0, transition_width=500, fs=fs_5cm) # boost where the ear is most sensitive
set_band_gain_window(x=X_5cm, f1=90, f2=140, inner_gain_db=-np.inf, outer_gain_db=0, transition_width=25, fs=fs_5cm) # attenuate plosives
clean_5cm = np.real(ifft(X_5cm))

wavfile.write("enhanced_speech_5cm.wav", fs_5cm, clean_5cm.astype(np.float32))

X_1m = fft(signal_1m)
set_band_gain_window(X_1m, f1=94.5, f2=10960, inner_gain_db=0, outer_gain_db=-np.inf, transition_width=50, fs=fs_1m)
X_1m_denoised = X_1m.copy()
set_band_gain_window(x=X_1m, f1=2000, f2=6000, inner_gain_db=3.5, outer_gain_db=0, transition_width=500, fs=fs_1m) # boost where the ear is most sensitive
set_band_gain_window(x=X_1m, f1=100, f2=300, inner_gain_db=3.5, outer_gain_db=0, transition_width=50, fs=fs_1m) # bass boost
clean_1m = np.real(ifft(X_1m))

# make 1m louder to match 5cm
rms_5cm = np.sqrt(np.mean(clean_5cm**2)) # average vol of 5cm
rms_1m = np.sqrt(np.mean(clean_1m**2)) # average vol of 1m

equalise = rms_5cm / rms_1m
clean_1m = clean_1m * equalise

wavfile.write("enhanced_speech_1m.wav", fs_1m, clean_1m.astype(np.float32))

# used for checking
denoised_5cm = np.real(ifft(X_5cm_denoised))
noise_removed_5cm = signal_5cm - denoised_5cm
wavfile.write("team23_5cm_removed.wav", fs_5cm, (noise_removed_5cm * 10).astype(np.float32))

denoised_1m = np.real(ifft(X_1m_denoised))
noise_removed_1m = signal_1m - denoised_1m
wavfile.write("team23_1m_removed.wav", fs_1m, (noise_removed_1m * 10).astype(np.float32))