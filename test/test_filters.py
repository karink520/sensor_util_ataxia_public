from sensor_util.filters import (lowpass_filter, highpass_filter, wavelet_denoising)

# These are super basic tests that just make sure the code still runs

import numpy as np
import pandas as pd

def test_lowpass_filter():
    t = np.linspace(0, 10, 1000)
    data = np.sin(2*np.pi*t) + 5*np.cos(20*np.pi*t) 
    fs = 100
    cutoff = 20
    lowpass_filter(data, cutoff, fs, filter_type="butterworth", order=5, padlen=0)
    lowpass_filter(data, cutoff, fs, filter_type="bessel", order=4, padlen=0)


def test_highpass_filter():
    t = np.linspace(0, 10, 1000)
    data = np.sin(2*np.pi*t) + 5*np.cos(20*np.pi*t) 
    fs = 100
    cutoff = 20
    highpass_filter(data, cutoff, fs, filter_type="butterworth", order=5, padlen=0)
    highpass_filter(data, cutoff, fs, filter_type="bessel", order=4, padlen=0)


def test_wavelet_denoising():
    t = np.linspace(0, 10, 1000)
    data = np.sin(2*np.pi*t) + 5*np.cos(20*np.pi*t) 
    wavelet_denoising(data, wavelet_name="sym4", threshold=0.1)
    data = pd.Series(np.sin(2*np.pi*t) + 5*np.cos(20*np.pi*t)) 
    wavelet_denoising(data, wavelet_name="sym4", threshold=0.05, level=2)

