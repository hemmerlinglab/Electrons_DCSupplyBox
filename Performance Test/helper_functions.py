import pandas as pd
import numpy as np

def find_regulator_boundaries(x, y):
    """
    Find the boundaries between the 'following' region and the 'regulation' region
    for a voltage regulator transfer curve.

    Args:
        x (np.ndarray): Input voltage values (monotonically increasing or decreasing).
        y (np.ndarray): Output voltage values (same length as x).

    Returns:
        boundary1 (float): Voltage at which the regulator transitions from 'following' 
                           (output ≈ input) to 'regulating' (output nearly constant).
        boundary2 (float): Voltage at which 'regulation' is confidently established.

    Method:
        - Compute the slope (dy/dx) between each pair of adjacent points.
        - The 'following' region is where the slope is close to 1 (output tracks input).
        - The 'regulation' region is where the slope is close to 0 (output is flat).
        - The boundary is set at the midpoint between the last 'follow' index and first 'regulation' index.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    slope = dy / dx

    follow_threshold = 0.99         # Slope ≥ 0.99 considered 'following'
    regulation_threshold = 0.01     # Slope ≤ 0.01 considered 'regulating'

    is_follow = slope >= follow_threshold
    is_regulation = slope <= regulation_threshold

    # Find the last point that is 'following'
    last_follow_index = np.max(np.where(is_follow))
    # Find the first point that is 'regulating'
    first_regulation_index = np.min(np.where(is_regulation))

    # Compute boundary voltages as the midpoint between adjacent x values
    boundary1 = (x[last_follow_index] + x[last_follow_index + 1]) / 2
    boundary2 = (x[first_regulation_index] + x[first_regulation_index + 1]) / 2

    return boundary1, boundary2

def compute_r2(x, y, coef):
    """
    Compute the coefficient of determination (R^2) for a linear fit y = kx + b.

    Args:
        x (np.ndarray): Input (independent variable).
        y (np.ndarray): Output (dependent variable).
        coef (array-like): Coefficients [k, b] from np.polyfit.

    Returns:
        r2 (float): R^2 value, representing the fraction of variance explained by the model.
    """
    k, b = coef
    y_pred = k * x + b                          # Fitted/predicted values
    ss_res = np.sum((y - y_pred) ** 2)          # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)      # Total sum of squares
    return 1 - ss_res / ss_tot                  # R^2 formula


def save_fit_results_csv(channels, ampdata, filename):
    """
    Save all fitting parameters and R^2 values to a CSV file.
    Columns: Channel, Fit Type, k (slope), b (offset), R2
    """
    rows = []
    fit_types = [
        ("fit_input_artiq", "Input→Artiq"),
        ("fit_artiq_amp", "Artiq→Amp"),
        ("fit_input_amp", "Input→Amp"),
    ]
    for ch in channels:
        for fitkey, fitname in fit_types:
            k, b = ampdata[ch][fitkey]["coeff"]
            r2 = ampdata[ch][fitkey]["r2"]
            rows.append({
                "Channel": ch,
                "Fit": fitname,
                "k": k,
                "b": b,
                "R2": r2,
            })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

def save_fit_results_txt(channels, ampdata, filename):
    """
    Save all fitting parameters and R^2 values to a formatted text file for human reading.
    """
    fit_types = [
        ("fit_input_artiq", "Input→Artiq"),
        ("fit_artiq_amp", "Artiq→Amp"),
        ("fit_input_amp", "Input→Amp"),
    ]
    with open(filename, "w") as f:
        f.write("Fitting Parameters by Channel\n")
        f.write("="*60 + "\n\n")
        for ch in channels:
            f.write(f"Channel {ch}\n")
            for fitkey, fitname in fit_types:
                k, b = ampdata[ch][fitkey]["coeff"]
                r2 = ampdata[ch][fitkey]["r2"]
                f.write(f"  {fitname}:\n")
                f.write(f"    k (slope): {k:.6f}\n")
                f.write(f"    b (offset): {b:.6f}\n")
                f.write(f"    R^2: {r2:.6f}\n")
            f.write("\n")
