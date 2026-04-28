import numpy as np

def normalise(flux):
    """
    Normalise a flux array so it hovers around 1.0

    A transit dip then becomes something like 0.9985 instead of 63,500
    which makes it easy to measure and compare across different stars.

    Parameters: 
        flux: numpy array of raw flux values

    Returns:
        numpy array of normalised flux values
    """

    median = np.median(flux)
    normalised_flux = flux / median
    return normalised_flux

def remove_outliers(time, flux, sigma=3.0):
    """
    Remove data points that are statistically implausible.
    
    We use sigma-clipping: Any point more than `sigma` 
    standard deviations from the median gets removed. 
    Default is 3-sigma (covers 99.7% of
    normal variation, so we only cut genuine outliers).
    
    Parameters:
        time  : array of time values
        flux  : array of normalised flux values
        sigma : how many standard deviations = outlier threshold
    
    Returns:
        cleaned time and flux arrays
    """
    median = np.median(flux)
    std = np.std(flux)
    
    # Build a mask
    mask = np.abs(flux - median) < sigma * std
    
    removed = len(flux) - np.sum(mask)
    print(f"Removed {removed} outliers ({removed/len(flux)*100:.2f}% of data)")
    
    return time[mask], flux[mask]

def flatten(time, flux, window_size=301):
    """
    window_size=301 — we look at 301 points surrounding each data point.
    At 30-minute cadence, 301 points = ~150 hours = ~6 days. Long enough to capture 
    slow stellar trends, short enough to not swallow transit signals 
    (which last only ~10 hours).

    Remove slow stellar trends by dividing out a smoothed version
    of the light curve. This is called 'detrending'.
    
    Method - sliding median filter:
    For each point, look at the `window_size` surrounding points,
    take their median, and use that as the 'expected' brightness.
    Dividing by this removes the slow wave while preserving
    short sharp dips (transits).
    
    window_size=301 means we look at ~150 points either side.
    At 30min cadence that's ~75 hours — long enough to smooth
    stellar trends, short enough to preserve transit signals
    (which last only a few hours).
    
    Parameters:
        time        : array of time values
        flux        : array of normalised flux values  
        window_size : must be odd number
    
    Returns:
        flattened flux array
    """
    if window_size % 2 == 0:
        window_size += 1  
    
    half = window_size // 2
    trend = np.zeros(len(flux))
    
    for i in range(len(flux)):
        # Define window boundaries 
        start = max(0, i - half)
        end   = min(len(flux), i + half + 1)
        
        # Median of surrounding points = local trend
        trend[i] = np.median(flux[start:end])
    
    # Divide out the trend — what remains is just the transit signals + noise
    flattened = flux / trend
    
    return flattened, trend