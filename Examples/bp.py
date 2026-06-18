"""
This is a basic backprojection example using the MIT Radar data (although you can use your own data as well). It loads the radar data, computes the range profiles, 
and then applies the backprojection algorithm to create an image. The resulting image is displayed using Matplotlib. 
You can adjust the radar parameters and backprojection settings as needed to see how they affect the resulting image.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


import Data
import Imaging


RESOLUTION = (100, 100) # pixels
MAG_SCALE = 10.0 # dB

def main():
    radar = Data.MITRadar(filename="SAR Data/UXOClass_020226_Grouplevi_experiment7.wav")
    # Change radar parameters here if needed, e.g.:
    # radar.H_RADAR_FT = 20.0
    # radar.DELTA_X = 0.5
    # 
    # Alternatively, just pass the range profiles, radar positions, and dr directly to Imaging.backproject() if you have them precomputed.

    rp, positions, dr = radar.compute_range_profiles()

    img, crossrange_axis, downrange_axis, _ = Imaging.backproject(
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
    peak = np.max(img)
    plt.figure(figsize=(4, 4))
    plt.imshow(
        img,
        cmap="jet",
        origin="lower",
        extent=(
            crossrange_axis[0] / radar.FT,
            crossrange_axis[-1] / radar.FT,
            downrange_axis[0] / radar.FT,
            downrange_axis[-1] / radar.FT
        ),
    vmin=peak - MAG_SCALE,
    vmax=peak
    )
    plt.colorbar()
    plt.xlabel(f"Cross range")
    plt.ylabel(f"Down range")
    plt.title("Backprojected Image")
    plt.show()

if __name__ == '__main__':
    main()