import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, gaussian_filter1d

import Data
import Imaging
import Autofocus


RESOLUTION = (100, 100)
DYNAMIC_RANGE_DB = 10.0

def main():
    radar = Data.MITRadar(filename="SAR Data/UXOClass_020226_Grouplevi_experiment1.wav")
    rp, positions, dr = radar.compute_range_profiles()
    base_parameters = {
        "crossrange": (-10 * radar.FT, 10 * radar.FT),
        "downrange": (0 * radar.FT, 25 * radar.FT),
        "resolution": RESOLUTION,
        "fstart": radar.FSTART,
        "fstop": radar.FSTOP,
        "phase_sign": 1.0,
        "output_db": True,
        "normalize_db": True,
        "flip_lr": False,
        "flip_ud": False,
        "transpose_output": False,
        "os_factor": radar.IMAGE_OVERSAMPLE,
        "dr": dr
    }
    base_img, crossrange_axis, downrange_axis, _ = Imaging.backproject(
        positions=positions,
        range_profiles=rp,
        **base_parameters
    )

    positions_err = positions.copy()

    long_error = 0.03
    lat_error = 0.02
    alt_error = 0.06

    positions_err[:, 0] += np.random.uniform(-lat_error / 2, lat_error / 2, positions_err.shape[0])
    positions_err[:, 1] += np.random.uniform(-long_error / 2, long_error / 2, positions_err.shape[0])
    positions_err[:, 2] += np.random.uniform(-alt_error / 2, alt_error / 2, positions_err.shape[0])

    error_img, _, _, _ = Imaging.backproject(
        positions=positions_err,
        range_profiles=rp,
        **base_parameters
    )

    mea_img, diffs, entropy, mea_params = Autofocus.parameterized_mea(
        positions=positions_err,
        range_profiles=rp,
        base_parameters=base_parameters,
        stop_entropy_diff=1e-5,
        num_iterations=1000,
        order=6,                                             # Highest degree polynomial, gives an order - 2 term polynomial because a_0 and a_1 are removed.
        steps=[10]*4,                                      # Must be same length as order - 2, e.g. steps=[.1] for a 3rd order polynomial.
        learning_rates=[50, 20, 1, .5],       # Must be same length as order - 2, e.g. learning_rates=[.1] for a 3rd order polynomial.
        poly_coeffs=[0]*4
    )

    pga_img, _, pga_phase_error = Autofocus.phase_gradient_autofocus(
        positions=positions_err,
        range_profiles=rp,
        base_parameters=base_parameters
    )

    # mea_pga_parameters = base_parameters.copy()
    # mea_pga_parameters.update(mea_params)

    # mea_pga_focused, _, mea_pga_phase_error = Autofocus.phase_gradient_autofocus(
    #     positions=positions_err,
    #     range_profiles=rp,
    #     base_parameters=mea_pga_parameters,
    #     std_coeff=2,
    #     neighborhood_size=7
    # )


    peak = np.max(base_img)
    extent = (
        crossrange_axis[0] / radar.FT,
        crossrange_axis[-1] / radar.FT,
        downrange_axis[0] / radar.FT,
        downrange_axis[-1] / radar.FT
    )

    images = [
        (base_img, "Original"),
        (error_img, f"Motion Error ({long_error * 100:.1f}, {lat_error * 100:.1f}, {alt_error * 100:.1f}) cm"),
        (mea_img, "Parameterized MEA"),
        (pga_img, "PGA"),
        # (mea_pga_img, "MEA + PGA")
    ]

    fig, axs = plt.subplots(1, 4, figsize=(28, 5))

    for ax, (img, title) in zip(axs, images):
        im = ax.imshow(
            img,
            cmap="inferno",
            origin="lower",
            extent=extent,
            vmin=peak - DYNAMIC_RANGE_DB,
            vmax=peak
        )
        ax.set_xlabel("Cross range")
        ax.set_ylabel("Down range")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(pga_phase_error, label="PGA")
    # plt.plot(mea_pga_phase_error, label="MEA + PGA")
    plt.legend()
    plt.title("Estimated Phase Error")
    plt.xlabel("Aperture Index")
    plt.ylabel("Phase Error [rad]")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()