import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import find_regulator_boundaries, compute_r2, save_fit_results_csv, save_fit_results_txt

# ====== Basic configuration ======
# Define the physical layout of amplifier channels (4 rows x 5 columns)
channels_layout = [['1',  '2',  '3',  '4',   '5'],
                   ['6',  '7',  '8',  '9',  '10'],
                   ['11', '12', '13', '14', '15'],
                   ['16', '17', '18', '19', '20']]
channels = [ch for row in channels_layout for ch in row]
regulators = ['Neg', 'Pos']
Nrow = len(channels_layout)
Ncol = len(channels_layout[0])

# ====== Data reading functions ======
def read_reg(file):
    """
    Read regulator module data from a CSV file.
    Returns input and output voltage arrays.
    """
    df = pd.read_csv(file)
    input = np.array(df["Input Voltage (V)"])
    output = np.array(df[" Output Voltage (V)"])
    return input, output

def read_ch(file):
    """
    Read channel data (amplifier module) from a CSV file.
    Returns input, Artiq output, and amplifier output arrays.
    """
    df = pd.read_csv(file)
    input = np.array(df["Artiq Input (V)"])
    artiq = np.array(df[" Artiq Output (V)"])
    amp = np.array(df[" Board Output (V)"])
    return input, artiq, amp

# ====== Plotting Functions ======
def plot_amp_grid(
    channels_layout,
    ampdata,
    xkey,
    ykey,
    fitkey,
    plot_kwargs=None,
    scatter_kwargs=None
):
    """
    Plots a grid of amplifier data for each channel, showing linear fits and scatter data.

    Args:
        channels_layout: 2D list specifying the arrangement of channel names.
        ampdata: Dictionary of per-channel data and fit results.
        xkey: Key for x-axis data (str).
        ykey: Key for y-axis data (str).
        fitkey: Key for fit parameters dict (must contain 'coeff' and 'r2').
        plot_kwargs: Dict of additional keyword arguments for ax.plot().
        scatter_kwargs: Dict of additional keyword arguments for ax.scatter().
    Returns:
        (fig, axes): The matplotlib Figure and Axes array.
    """
    Nrow = len(channels_layout)
    Ncol = len(channels_layout[0])
    fig, axes = plt.subplots(Nrow, Ncol, figsize=(4*Ncol, 3*Nrow))
    axes = np.array(axes).reshape(Nrow, Ncol)  # Ensure axes is 2D even if Nrow/Ncol==1

    if plot_kwargs is None:
        plot_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}

    # Iterate through each channel and create the corresponding subplot
    for i, row in enumerate(channels_layout):
        for j, ch in enumerate(row):
            ax = axes[i][j]
            xdata = ampdata[ch][xkey]
            ydata = ampdata[ch][ykey]
            k, b = ampdata[ch][fitkey]["coeff"]
            r2 = ampdata[ch][fitkey]["r2"]

            # Generate fitted line data
            xfit = np.linspace(np.min(xdata), np.max(xdata), 100)
            yfit = k * xfit + b

            # Format the intercept (b) label based on plot context
            if ykey == "amp" and xkey == "artiq":
                b_label = f"{b*1000:.2f}mV"
            elif ykey == "artiq":
                b_label = f"{b*1000:.1f}mV"
            else:
                b_label = f"{b:.2f}V"

            ax.plot(xfit, yfit, label=f"k = {k:.2f}, b = {b_label}\n$r^2$={r2:.3f}", color="red", **plot_kwargs)
            ax.scatter(xdata, ydata, label=f"ch{ch}", marker="x", color="black", s=20, **scatter_kwargs)
            ax.legend()

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    return fig, axes

def plot_fit_params(channels, ampdata, fitkey, title_prefix):
    """
    Plots slope (k), offset (b), and (1-R^2) for all channels for a given fit.

    Args:
        channels: List of channel names.
        ampdata: Dictionary of per-channel data and fit results.
        fitkey: Which linear fit to visualize.
        title_prefix: Used as title prefix for each subplot.
    Returns:
        (fig, axes): The matplotlib Figure and Axes array.
    """
    x = np.arange(len(channels))
    xlabels = [f"ch{ch}" for ch in channels]
    k_list = []
    b_list = []
    r2_list = []

    for ch in channels:
        k, b = ampdata[ch][fitkey]["coeff"]
        r2 = ampdata[ch][fitkey]["r2"]
        k_list.append(k)
        b_list.append(b)
        r2_list.append(r2)
    b = np.array(b_list)
    r2 = np.array(r2_list)

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(channels)//2), 12), sharex=True)

    axes[0].plot(x, k_list, marker="o")
    axes[0].set_ylabel("k (slope)")
    axes[0].set_title(f"{title_prefix} Slope k by Channel")

    axes[1].plot(x, b*1000, marker="o")
    axes[1].set_ylabel("b (offset, mV)")
    axes[1].set_title(f"{title_prefix} Offset b by Channel")

    axes[2].plot(x, 1-r2, marker="o")
    axes[2].set_ylabel("$1-R^2$")
    axes[2].set_title(f"{title_prefix} $R^2$ by Channel")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Channel")
    axes[2].grid()

    # Set x-tick labels for all subplots
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45)

    plt.tight_layout()
    return fig, axes

# ====== Main Execution Block ======
if __name__ == "__main__":

    regdata = {}
    ampdata = {}

    # --- Regulator Module: Read and analyze each regulator type (Negative, Positive) ---
    for reg in regulators:
        regdata[reg] = {}
        input, output = read_reg("VoltageRegulatorModule" + reg + ".csv")
        boundary1, boundary2 = find_regulator_boundaries(input, output)
        regdata[reg]["input"] = input
        regdata[reg]["output"] = output
        regdata[reg]["boundaries"] = (boundary1, boundary2)

    # --- Amplifier Module: Read and analyze each channel ---
    for ch in channels:
        ampdata[ch] = {}
        input, artiq, amp = read_ch("ch" + ch + ".csv")
        # Fit: input→artiq, artiq→amp, input→amp
        coef_input_artiq = np.polyfit(input, artiq, deg=1)
        coef_artiq_amp = np.polyfit(artiq, amp, deg=1)
        coef_input_amp = np.polyfit(input, amp, deg=1)
        # Compute R^2 for each fit
        r2_input_artiq = compute_r2(input, artiq, coef_input_artiq)
        r2_artiq_amp = compute_r2(artiq, amp, coef_artiq_amp)
        r2_input_amp = compute_r2(input, amp, coef_input_amp)

        ampdata[ch]["input"] = input
        ampdata[ch]["artiq"] = artiq
        ampdata[ch]["amp"] = amp
        ampdata[ch]["fit_input_artiq"] = {"coeff": coef_input_artiq, "r2": r2_input_artiq}
        ampdata[ch]["fit_artiq_amp"] = {"coeff": coef_artiq_amp, "r2": r2_artiq_amp}
        ampdata[ch]["fit_input_amp"] = {"coeff": coef_input_amp, "r2": r2_input_amp}

    # --- Regulator Module Plotting: Regulating Region Only ---
    fig0, axes0 = plt.subplots(len(regulators), 1, figsize=(8, 6*len(regulators)))
    for i, reg in enumerate(regulators):
        # Only plot region where the regulator is actively regulating (beyond the threshold)
        mask = abs(regdata[reg]["input"]) > abs(regdata[reg]["boundaries"][1])
        x = regdata[reg]["input"][mask]
        y = regdata[reg]["output"][mask]
        avg_k = (y[-1] - y[0]) / (x[-1] - x[0])
        axes0[i].plot(x, y, label=f"Average Slope = {avg_k*100:.2f}%")
        axes0[i].set_xlabel("Input Voltage (V)")
        axes0[i].set_ylabel("Output Voltage (V)")
        axes0[i].set_title("Regulator " + reg + " Regulating Region")
        axes0[i].legend()
    fig0.savefig("Reg_Regulating.png", dpi=150)
    
    # --- Regulator Module Plotting: Full Performance Curve ---
    fig1, axes1 = plt.subplots(len(regulators), 1, figsize=(8, 6*len(regulators)))
    for i, reg in enumerate(regulators):
        axes1[i].plot(regdata[reg]["input"], regdata[reg]["output"], label=f"Regulating Threshold: {regdata[reg]['boundaries'][1]:.1f}")
        axes1[i].axvline(x=regdata[reg]["boundaries"][0], linestyle="--", linewidth=0.6, color='grey')
        axes1[i].axvline(x=regdata[reg]["boundaries"][1], linestyle="--", linewidth=0.6, color='grey')
        axes1[i].set_xlabel("Input Voltage (V)")
        axes1[i].set_ylabel("Output Voltage (V)")
        axes1[i].set_title("Regulator " + reg + " Full Performance")
        axes1[i].legend()
    fig1.savefig("Reg_Full_Performance.png", dpi=150)

    # --- Artiq (Zotino) Output-Input Transfer Function (Per Channel) ---
    fig2, _ = plot_amp_grid(channels_layout, ampdata, xkey="input", ykey="artiq", fitkey="fit_input_artiq")
    fig2.suptitle("Artiq Zotino Output-Input")
    fig2.supxlabel("Zotino Control Voltage (V)")
    fig2.supylabel("Zotino Output Voltage (V)")
    fig2.savefig("Zotino_IO.png", dpi=150)

    # --- Amplifier Board Transfer Function (Per Channel) ---
    fig3, _ = plot_amp_grid(channels_layout, ampdata, xkey="artiq", ykey="amp", fitkey="fit_artiq_amp")
    fig3.suptitle("Amplifier Board Performance")
    fig3.supxlabel("Zotino Output Voltage (V)")
    fig3.supylabel("Amplifier Output Voltage (V)")
    fig3.savefig("Amplifier_IO.png", dpi=150)

    # --- Overall System Output-Input (Per Channel) ---
    fig4, _ = plot_amp_grid(channels_layout, ampdata, xkey="input", ykey="amp", fitkey="fit_input_amp")
    fig4.suptitle("System Output-Input")
    fig4.supxlabel("Zotino Control Voltage (V)")
    fig4.supylabel("Amplifier Output Voltage (V)")
    fig4.savefig("System_IO.png", dpi=150)

    # --- Plot Linear Fit Parameters (Slope, Offset, R^2) for All Channels ---
    fig5, _ = plot_fit_params(channels, ampdata, "fit_input_artiq", "Input→Artiq")
    fig5.savefig("Fit_Params_input_artiq.png", dpi=150)
    fig6, _ = plot_fit_params(channels, ampdata, "fit_artiq_amp", "Artiq→Amp")
    fig6.savefig("Fit_Params_artiq_amp.png", dpi=150)
    fig7, _ = plot_fit_params(channels, ampdata, "fit_input_amp", "Input→Amp")
    fig7.savefig("Fit_Params_input_amp.png", dpi=150)

    save_fit_results_csv(channels, ampdata, "fit_parameters.csv")
    save_fit_results_txt(channels, ampdata, "fit_parameters.txt")
    #plt.show()
