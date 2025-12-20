import numpy as np
import matplotlib.pyplot as plt

def load_txt(file_name):
    """Load real and predicted values from a txt file."""
    data = np.loadtxt(file_name)
    real = data[:, 0]
    predicted = data[:, 1]
    return real, predicted

def get_abs_error_cdf(real, predicted, max_points=5000):
    """Compute filtered and downsampled absolute error CDF."""
    abs_error = np.abs(predicted - real)

    # Filter out large errors
    abs_error = abs_error[abs_error <= 100]
    abs_error_sorted = np.sort(abs_error)
    cdf = np.linspace(0, 1, len(abs_error_sorted))

    # Downsample for plotting
    if len(abs_error_sorted) > max_points:
        idx = np.linspace(0, len(abs_error_sorted) - 1, max_points, dtype=int)
        abs_error_sorted = abs_error_sorted[idx]
        cdf = cdf[idx]

    return abs_error_sorted, cdf

def plot_abs_error_cdf_dual():
    """
    Plot two side-by-side CDFs (Prefill and Decode),
    each comparing BurstGPT and Azure.
    """
    # === Load data ===
    real_prefill_burst, pred_prefill_burst = load_txt("prefill_burstgpt.txt")
    real_prefill_azure, pred_prefill_azure = load_txt("prefill_azure.txt")
    real_decode_burst, pred_decode_burst = load_txt("decode_burstgpt.txt")
    real_decode_azure, pred_decode_azure = load_txt("decode_azure.txt")

    # === Compute CDFs ===
    prefill_burst, cdf_prefill_burst = get_abs_error_cdf(real_prefill_burst, pred_prefill_burst)
    prefill_azure, cdf_prefill_azure = get_abs_error_cdf(real_prefill_azure, pred_prefill_azure)
    decode_burst, cdf_decode_burst = get_abs_error_cdf(real_decode_burst, pred_decode_burst)
    decode_azure, cdf_decode_azure = get_abs_error_cdf(real_decode_azure, pred_decode_azure)

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    # Left: Prefill
    axes[0].plot(prefill_burst, cdf_prefill_burst, label="BurstGPT", alpha=0.8)
    axes[0].plot(prefill_azure, cdf_prefill_azure, label="Azure", alpha=0.8)
    axes[0].set_xlabel("TPFT Absolute Error")
    axes[0].set_ylabel("CDF")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    # axes[0].legend()

    # Right: Decode
    axes[1].plot(decode_burst, cdf_decode_burst, label="BurstGPT", alpha=0.8)
    axes[1].plot(decode_azure, cdf_decode_azure, label="Azure", alpha=0.8)
    axes[1].set_xlabel("TPDT Absolute Error")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    # axes[1].legend()

    # ✅ Combine legends from both axes into one
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=3.0)
    plt.savefig("abs_error_cdf_comparison.pdf")
    plt.close()

# === Run ===
if __name__ == "__main__":
    plot_abs_error_cdf_dual()
