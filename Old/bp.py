import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from PIL import Image
from numba import njit, prange
from Imaging import *

Tp = 5e-3
ir_thresh = 0.20
min_ir_sep_s = 0.025

delta_x = 0.011
h_radar_ft = 6
h_radar_m = h_radar_ft * 0.3048

c = 3e8
fstart = 2280e6
fstop = 2580e6
B = fstop - fstart
fc = (fstart + fstop) / 2

range_gate_ft = 8.0
range_gate_fade_ft = 1.0

debug_zpad = 8192
debug_max_bin = 700
debug_dynamic_range = 20


def get_range_profiles(filename):
    FS, Y = wavfile.read(filename)
    Y = Y.astype(np.float32)

    if Y.ndim == 1:
        raise ValueError("WAV is mono. Expected multichannel WAV.")

    if np.max(np.abs(Y)) > 1:
        Y = Y / np.max(np.abs(Y))

    radar_raw = Y[:, 1]
    sync_raw = Y[:, 2]
    ir_raw = Y[:, 3]

    N = int(round(Tp * FS))

    sync_bool = (sync_raw > 0).astype(np.int8)
    ir_bool = (np.abs(ir_raw) > ir_thresh).astype(np.int8)

    sync_starts = np.where(np.diff(sync_bool) == 1)[0] + 1
    ir_edges = np.where(np.diff(ir_bool) == 1)[0] + 1

    min_dist = int(FS * min_ir_sep_s)
    valid_ir = []
    last_c = -min_dist

    for edge in ir_edges:
        if edge > last_c + min_dist:
            valid_ir.append(edge)
            last_c = edge

    valid_ir = np.array(valid_ir)

    sif_time = []
    used_ir = []
    used_sync = []

    for edge in valid_ir:
        idx_s = np.argmin(np.abs(sync_starts - edge))
        bst = int(sync_starts[idx_s])

        if bst + N <= len(radar_raw):
            sif_time.append(radar_raw[bst:bst + N])
            used_ir.append(edge)
            used_sync.append(bst)

    sif_time = np.array(sif_time)
    used_ir = np.array(used_ir)
    used_sync = np.array(used_sync)

    if len(sif_time) == 0:
        raise ValueError("No valid pulses found.")

    sif_time = sif_time - np.mean(sif_time, axis=1, keepdims=True)

    win = np.hanning(N)
    range_profiles_complex = np.fft.fft(sif_time * win[None, :], axis=1)
    # range_profiles_complex = np.fft.ifft(sif_time, axis=1)

    range_res_m = c / (2 * B)
    gate_m = range_gate_ft * 0.3048
    fade_m = range_gate_fade_ft * 0.3048

    stop_bins = int(round(gate_m / range_res_m))
    fade_bins = int(round(fade_m / range_res_m))

    mask = np.ones(range_profiles_complex.shape[1], dtype=np.float32)

    if stop_bins < len(mask) // 2:
        mask[:stop_bins] = 0
        mask[-stop_bins:] = 0

        if fade_bins > 0:
            ramp_up = np.linspace(0, 1, fade_bins)
            ramp_down = np.linspace(1, 0, fade_bins)

            a0 = stop_bins
            a1 = min(stop_bins + fade_bins, len(mask))
            mask[a0:a1] = ramp_up[:a1 - a0]

            b1 = len(mask) - stop_bins
            b0 = max(0, b1 - fade_bins)
            mask[b0:b1] = ramp_down[-(b1 - b0):]

    range_profiles_complex = range_profiles_complex * mask[None, :]

    q_debug = np.fft.ifft(sif_time, n=debug_zpad, axis=1)

    debug_mask = np.ones(debug_zpad, dtype=np.float32)
    scale = debug_zpad / range_profiles_complex.shape[1]
    stop_debug = int(round(stop_bins * scale))
    fade_debug = int(round(fade_bins * scale))

    if stop_debug < debug_zpad // 2:
        debug_mask[:stop_debug] = 0
        debug_mask[-stop_debug:] = 0

        if fade_debug > 0:
            ramp_up = np.linspace(0, 1, fade_debug)
            ramp_down = np.linspace(1, 0, fade_debug)

            a0 = stop_debug
            a1 = min(stop_debug + fade_debug, debug_zpad)
            debug_mask[a0:a1] = ramp_up[:a1 - a0]

            b1 = debug_zpad - stop_debug
            b0 = max(0, b1 - fade_debug)
            debug_mask[b0:b1] = ramp_down[-(b1 - b0):]

    q_debug = q_debug * debug_mask[None, :]

    range_profiles_mag = np.abs(q_debug)
    range_profiles_db = 20 * np.log10(range_profiles_mag + 1e-12)

    vmax = np.max(range_profiles_db)
    vmin = vmax - debug_dynamic_range

    num_profiles = sif_time.shape[0]

    x = np.arange(num_profiles) * delta_x
    x = x - np.mean(x)
    y = np.zeros(num_profiles)
    z = np.full(num_profiles, h_radar_m)

    radar_positions = np.column_stack([x, y, z])

    print("Pulses:", len(ir_edges))
    print("FS:", FS)
    print("N:", N)
    print("num profiles:", num_profiles)
    print("profile length:", q_debug.shape[1])
    print("range resolution m:", range_res_m)
    print("range gate bins:", stop_bins)
    print("fade bins:", fade_bins)
    print("radar_positions shape:", radar_positions.shape)
    print("first position:", radar_positions[0])
    print("last position:", radar_positions[-1])
    print("aperture width m:", np.ptp(radar_positions[:, 0]))

    fig, ax = plt.subplots(1, 3, figsize=(22, 7))
    ax = ax[0]

    im = ax.imshow(
        range_profiles_db.T,
        cmap="jet",
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )

    peak_bins = np.argmax(range_profiles_mag, axis=1)

    ax.scatter(
        np.arange(len(peak_bins)),
        peak_bins,
        s=8,
        c="purple",
        marker="."
    )

    ax.set_ylim(0, debug_max_bin)
    ax.set_ylabel("Bin")
    ax.set_xlabel("Pulse")
    ax.set_title("Range Compressed History with Peak Bins")

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    ticks = np.linspace(0, len(radar_positions) - 1, 6)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{radar_positions[int(t), 0]:.2f}" for t in ticks])
    ax2.set_xlabel("Radar x-position (m)")

    plt.colorbar(im, ax=ax, label="dB")
    # plt.show()

    range_profiles = range_profiles_complex
    return range_profiles, radar_positions, fig, ax




range_profiles, radar_positions, fig, ax = get_range_profiles("/Users/levifarinas/Library/Mobile Documents/com~apple~CloudDocs/Projects/SAR Backprojection/SAR Data/UXOClass_020226_Groupandrew_experiment1_caponly.wav")

rma_img, rma_cross_axis, rma_down_axis = rma(
    range_profiles,
    radar_positions
)

img, crossrange_axis, downrange_axis = bp(
    radar_positions,
    range_profiles,
    crossrange=(rma_cross_axis[0], rma_cross_axis[-1]),
    downrange=(rma_down_axis[0], rma_down_axis[-1]),
    resolution=(rma_img.shape[1], rma_img.shape[0])
)

ax_bp = fig.axes[1]

im2 = ax_bp.imshow(
    img,
    cmap="jet",
    origin="lower",
    extent=[downrange_axis[0], downrange_axis[-1], crossrange_axis[0], crossrange_axis[-1]],
    vmin=-40,
    vmax=0
)

ax_bp.set_xlabel("Cross Range")
ax_bp.set_ylabel("Down Range")
ax_bp.set_title("Backprojection Image")

plt.colorbar(im2, ax=ax_bp, label="dB")

ax_rma = fig.axes[2]

peak = np.max(rma_img)

im3 = ax_rma.imshow(
    rma_img,
    cmap="jet",
    origin="lower",
    extent=[rma_cross_axis[0], rma_cross_axis[-1], rma_down_axis[0], rma_down_axis[-1]],
    vmin=peak - 40,
    vmax=peak
)

ax_rma.set_xlabel("Cross Range")
ax_rma.set_ylabel("Down Range")
ax_rma.set_title("RMA Image")

plt.colorbar(im3, ax=ax_rma, label="dB")

plt.tight_layout()
plt.show()