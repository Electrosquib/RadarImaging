"""
This example builds off of bp.py, but intentionally introduces a random motion error to the radar positions to demonstrate how it affects the resulting image. 
The motion error is applied as a random perturbation to the radar positions, simulating the effect of platform instability or other sources of motion during data collection. 
By comparing the resulting image with and without the motion error, you can see how it degrades the image quality and causes blurring or distortion. 
You can adjust the magnitude of the motion error to see how it affects the image further.

The motion error is applied by directly modifying the radar positions before passing them to the backprojection function. The positions are stored in a N x 3 array, 
where each column corresponds to a radar position in 3D space (x, y, z).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


import Data
import Imaging


RESOLUTION = (100, 100) # pixels
DYNAMIC_RANGE_DB = 10.0 # dB

def main():
    radar = Data.MITRadar(filename="SAR Data/UXOClass_020226_Grouplevi_experiment7.wav")
    # Change radar parameters here if needed, e.g.:
    # radar.H_RADAR_FT = 20.0
    # radar.DELTA_X = 0.5
    # 
    # Alternatively, just pass the range profiles, radar positions, and dr directly to Imaging.backproject() if you have them precomputed.

    rp, positions, dr = radar.compute_range_profiles()

    # Introduce random motion error to the radar positions
    long_error = .03 # meters
    lat_error = .02 # meters
    alt_error = .06 # meters

    img0, crossrange_axis, downrange_axis, _ = Imaging.backproject(
        positions=positions,
        range_profiles=rp,
        crossrange=(-10 * radar.FT, 10 * radar.FT),
        downrange=(0 * radar.FT, 25 * radar.FT),
        resolution=RESOLUTION,
        fstart=radar.FSTART,
        fstop=radar.FSTOP,
        phase_sign=1.0,
        output_db=True,
        normalize_db=True,
        flip_lr=False,
        flip_ud=False,
        transpose_output=False,
        os_factor=radar.IMAGE_OVERSAMPLE,
        dr=dr,
        x_scale=1.0,
        x_offset=0.0,
        y_offset=0.0,
        z_offset=0.0,
        cable_off_ft=radar.CALIBRATION_OFFSET)
    peak = np.max(img0) 

    positions[:, 0] += np.random.uniform(-lat_error / 2, lat_error / 2, positions.shape[0])
    positions[:, 1] += np.random.uniform(-long_error / 2, long_error / 2, positions.shape[0])
    positions[:, 2] += np.random.uniform(-alt_error / 2, alt_error / 2, positions.shape[0])

    img1, crossrange_axis, downrange_axis, _ = Imaging.backproject(
        positions=positions,
        range_profiles=rp,
        crossrange=(-10 * radar.FT, 10 * radar.FT),
        downrange=(0 * radar.FT, 25 * radar.FT),
        resolution=RESOLUTION,
        fstart=radar.FSTART,
        fstop=radar.FSTOP,
        phase_sign=1.0,
        output_db=True,
        normalize_db=True,
        flip_lr=False,
        flip_ud=False,
        transpose_output=False,
        os_factor=radar.IMAGE_OVERSAMPLE,
        dr=dr,
        x_scale=1.0,
        x_offset=0.0,
        y_offset=0.0,
        z_offset=0.0,
        cable_off_ft=radar.CALIBRATION_OFFSET)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(
        img0,
        cmap="jet",
        origin="lower",
        extent=(
            crossrange_axis[0] / radar.FT,
            crossrange_axis[-1] / radar.FT,
            downrange_axis[0] / radar.FT,
            downrange_axis[-1] / radar.FT
        ),
    vmin=peak - DYNAMIC_RANGE_DB,
    vmax=peak
    )
    axs[0].set_xlabel(f"Cross range")
    axs[0].set_ylabel(f"Down range")
    axs[0].set_title("No Motion Error")
    fig.colorbar(im0, ax=axs[0])


    im1 = axs[1].imshow(
        img1,
        cmap="jet",
        origin="lower",
        extent=(
            crossrange_axis[0] / radar.FT,
            crossrange_axis[-1] / radar.FT,
            downrange_axis[0] / radar.FT,
            downrange_axis[-1] / radar.FT
        ),
    vmin=peak - DYNAMIC_RANGE_DB,
    vmax=peak
    )
    axs[1].set_xlabel(f"Cross range")
    axs[1].set_ylabel(f"Down range")
    axs[1].set_title(f"With Motion Error ({long_error*100}, {lat_error*100}, {alt_error*100}) cm")

    fig.colorbar(im1, ax=axs[1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()