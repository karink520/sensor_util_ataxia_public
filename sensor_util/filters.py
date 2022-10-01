from scipy.signal import butter, filtfilt, bessel
import pywt
import pandas as pd


def lowpass_filter(data, cutoff, fs, filter_type="butterworth", order=5, padlen=0):
    """Applies a lowpass filter to data and returns the filtered data

    Parameters
    ----------
    data : array_like
        The array of data to be filtered
    cutoff : float
        cutoff frequency for filter
    fs: float
        The sampling frequency used in creating `data`
    type: {`butterworth`, `bessel`}
        Type of filter to use. Butterworth has maximally flat frequency response, Bessel better preserves waveshape in passband
    order: int
        The order of the filter
    padlen:
        The number of elements by which to extend `data` at both ends of
        `axis` before applying the filter.  This value must be less than
        `data.shape[axis] - 1`.

    Returns
    -------
    filtered_data : ndarray
        lowpass filtered data with same shape as `data`
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if filter_type == "butterworth":
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
    elif filter_type == "bessel":
        b, a = bessel(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data, padlen=padlen)
    return filtered_data


def highpass_filter(data, cutoff, fs, filter_type="butterworth", order=10, padlen=0):
    """Applies a highpass filter to data and returns the filtered data

    Parameters
    ----------
    data : array_like
        The array of data to be filtered
    cutoff : float
        cutoff frequency for filter
    fs: float
        The sampling frequency used in creating `data`
    type: {`butterworth`, `bessel`}
        Type of filter to use. Butterworth: maximally flat frequencry response; Bessel: better preserves waveshape in passband
    order: int
        The order of the filter
    padlen:
        The number of elements by which to extend `data` at both ends of
        `axis` before applying the filter.  This value must be less than
        `data.shape[axis] - 1`.

    Returns
    -------
    filtered_data : ndarray
        highpass filtered data with same shape as `data`
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if filter_type == "butterworth":
        b, a = butter(order, normal_cutoff, btype="highpass", analog=False)
    elif filter_type == "bessel":
        b, a = bessel(order, normal_cutoff, btype="highpass", analog=False)
    filtered_data = filtfilt(b, a, data, padlen=padlen)
    return filtered_data


def wavelet_denoising(data, wavelet_name="sym4", threshold=0.04, level=None):
    """Applies wavelet denoising to a 1-d input, transforming, trimming small wavelet coefficients, inverting the transformation.

    Parameters
    ----------
    data : array_like
        The array of data to be filtered.  Assumed to be one-dimensional
    wavelet_name : str
        The particular wavelet to be used for the denoising.  See pywt.wavelist() for options
    threshold: float
        coefficients below this threshhold will be trimmed
    level: int, optional.
        decomposition level for the multilevel discrete wavelet transform.  If level is None, then it
        will be calculated using pywt.dwt_max_level()

    Returns
    -------
    datarec : ndarray
        1-D array with reconstructed signal from filtered coefficients
    """
    if isinstance(data, pd.Series):
        flattened_data = data.to_numpy()
        flattened_data = list(flattened_data.flat)
    else:
        flattened_data = list(data.flat)

    w = pywt.Wavelet(wavelet_name)
    maxlev = pywt.dwt_max_level(len(flattened_data), w.dec_len)
    if level is not None:
        maxlev = level
    coeffs = pywt.wavedec(flattened_data, wavelet_name, level=maxlev)

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
    datarec = pywt.waverec(coeffs, wavelet_name)
    return datarec
