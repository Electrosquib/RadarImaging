import numpy as np
from scipy.ndimage import maximum_filter
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import matplotlib.pyplot as plt

from Imaging import backproject as bp

def entropy_score(img):
    mag2 = np.abs(img) ** 2
    p = mag2 / (np.sum(mag2) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

def find_gradient(
    range_profiles, 
    positions,
    base_parameters,
    tuning_parameters={
        "x_offset": 0,
        "y_offset": 0,
        "z_offset": 0,
        "cable_off_ft": 0
    },
    steps={
        "x_offset": .1,
        "y_offset": .1,
        "z_offset": .1,
        "cable_off_ft": .1
    },
    base_entropy=None
    ):
    gradients = {}
    if base_entropy is None:
        img, c, d, _ = bp(positions=positions, range_profiles=range_profiles, **base_parameters, **tuning_parameters)
        base_entropy = entropy_score(img)
    for param in tuning_parameters.keys():
        tweaked_params = tuning_parameters.copy()
        tweaked_params[param] += steps[param]
        img, c, d, _ = bp(positions=positions, range_profiles=range_profiles, **base_parameters, **tweaked_params)
        gradients[param] = (entropy_score(img) - base_entropy) / steps[param]
    return gradients


def taylor(poly_coeffs, rp_len):
    """
    Returns Taylor Polynomial starting from n=2
    """
    eta = np.linspace(-1, 1, rp_len)
    poly = np.zeros(rp_len)
    for n, a_n in enumerate(poly_coeffs):
        poly += a_n * eta ** (n+2)
    return poly

def find_gradient_phase_error(
    range_profiles, 
    positions,
    base_parameters,
    steps=[],           # Must be same length as order - 2, e.g. steps=[.1] for a 3rd order polynomial
    poly_coeffs=[],       # Coefficients of Taylor polynomial starting from a_2 ... (a_0, a_1 are constant phase and spatial shift, useless).
    base_entropy=None,
    coeff_mask=None    # So you can do coordinate descent by only updating one coefficient at a time, e.g. coeff_mask=[1, 0, 0, 0] to only update a_2.
    ):
    if not len(poly_coeffs): raise ValueError("poly_coeffs must be defined.")
    if not len(steps): raise ValueError("steps must be defined.")
    gradients = [0] * len(poly_coeffs)
    if base_entropy is None:
        phase_error = taylor(poly_coeffs=poly_coeffs, rp_len=positions.shape[0])
        img, c, d, _ = bp(positions=positions, range_profiles=range_profiles, phase_error=phase_error, **base_parameters)
        base_entropy = entropy_score(img)
    active_coeffs = range(len(poly_coeffs)) if coeff_mask is None else [i for i, m in enumerate(coeff_mask) if m]
    for n in active_coeffs:
        tweaked_coeffs = poly_coeffs.copy()
        tweaked_coeffs[n] += steps[n]
        phase_error = taylor(poly_coeffs=tweaked_coeffs, rp_len=positions.shape[0])
        img, c, d, _ = bp(positions=positions, range_profiles=range_profiles, phase_error=phase_error, **base_parameters)
        gradients[n] = (entropy_score(img) - base_entropy) / steps[n]
    return gradients

def minimum_entropy_autofocus(
    range_profiles, 
    positions,
    base_parameters,
    stop_entropy_diff=None,
    num_iterations=5,
    learning_rates={
        "x_scale": 1e-3,
        "x_offset": 1e-3,
        "y_offset": 1e-3,
        "z_offset": 1e-3,
        "cable_off_ft": 1e-2
    },
    tuning_parameters={
        "x_scale": 1,
        "x_offset": 0,
        "y_offset": 0,
        "z_offset": 0,
        "cable_off_ft": 0
    },
    steps={
        "x_scale": 1e-1,
        "x_offset": 1e-1,
        "y_offset": 1e-1,
        "z_offset": 1e-1,
        "cable_off_ft": 1e-1
    }):
    entropy_diffs = []
    for i in range(num_iterations):
        img, c, d, _ = bp(positions=positions, range_profiles=range_profiles, **base_parameters, **tuning_parameters)
        base_entropy = entropy_score(img)
        gradients = find_gradient(range_profiles=range_profiles, positions=positions, base_parameters=base_parameters, tuning_parameters=tuning_parameters, steps=steps, base_entropy=base_entropy)
        new_params = {k: -gradients[k] * learning_rates[k] for k in gradients}
        img, c, d, _ = bp(positions=positions, range_profiles=range_profiles, **base_parameters, **new_params)
        new_entropy = entropy_score(img)
        entropy_diffs.append(new_entropy-base_entropy)
    return img, entropy_diffs, new_entropy, new_params


def parameterized_mea(
    range_profiles, 
    positions,
    base_parameters,
    stop_entropy_diff=5e-5,
    num_iterations=1000,
    order=6,           # Highest degree polynomial, gives an order - 2 term polynomial because a_0 and a_1 are removed.
    steps=[1e-3]*4,           # Must be same length as order - 2, e.g. steps=[.1] for a 3rd order polynomial.
    learning_rates=[1e-3]*4,  # Must be same length as order - 2, e.g. learning_rates=[.1] for a 3rd order polynomial.
    poly_coeffs=[0]*4,
    print_iterations_mod=50,
    coordinate_descent=False
):
    """
    Performs polynomial phase error MEA by finding the coefficients of a polynomial that models the phase error across the aperture. 
    The polynomial starts from n=2 since n=0 is a constant phase error that doesn't affect the image, 
    and n=1 is a linear phase error that corresponds to
    a spatial shift in the image, which can be easily corrected after the fact if the image is already well-focused 
    (e.g. by finding the peak and shifting it to the correct location).
    It uses gradient descent optimization to find the coefficients that minimize the entropy of the image, which is a common metric for image focus. 
    The gradients are found by tweaking each coefficient by a small amount and finding the change in entropy, i.e. a finite difference approximation of the gradient. 
    Coordinate descent can also be used by only updating one coefficient at a time, which can sometimes help with convergence since 
    the optimization problem is not necessarily convex.
    
    IMPORTANT: COORDINATE DESCENT IS NOT WORKING RN

    """
    if order is None or steps is None or learning_rates is None: raise ValueError("order, steps, and learning_rate must be defined!")
    entropies = []
    diff = float('inf')
    new_coeffs = np.array(poly_coeffs).astype(np.float64)
    learning_rates = np.array(learning_rates).astype(np.float64)
    poly_coeffs = np.array(poly_coeffs).astype(np.float64)
    count = 0
    best_entropy = float('inf')
    best_coeffs = new_coeffs.copy()
    best_img = None
    while diff > stop_entropy_diff:
        if coordinate_descent:
            active_coeff = count % len(new_coeffs)
            coeff_mask = [1 if i == active_coeff else 0 for i in range(len(new_coeffs))]
        else:
            active_coeff = None
            coeff_mask = None
                    
        phase_error = taylor(poly_coeffs=new_coeffs, rp_len=positions.shape[0])
        img, c, d, _ = bp(positions=positions, range_profiles=range_profiles, phase_error=phase_error, **base_parameters)
        base_entropy = entropy_score(img)
        gradients = find_gradient_phase_error(range_profiles=range_profiles, positions=positions, base_parameters=base_parameters, coeff_mask=coeff_mask, poly_coeffs=new_coeffs, steps=steps, base_entropy=base_entropy)
        gradients = np.array(gradients, dtype=np.float64)
        new_coeffs -= gradients * learning_rates
        img, c, d, _ = bp(positions=positions, range_profiles=range_profiles, phase_error=taylor(poly_coeffs=new_coeffs, rp_len=positions.shape[0]), **base_parameters)
        new_entropy = entropy_score(img)
        diff = abs(new_entropy - base_entropy)
        entropies.append(new_entropy)
        peak = np.max(img)
        if peak > 0:
            img = 20.0 * np.log10(img / peak + 1e-12)
        else:
            img = np.full_like(img, -120.0)
        img = np.nan_to_num(img, nan=-120.0, posinf=0.0, neginf=-120.0)
        count += 1
        if count % print_iterations_mod == 0:
            print(f"Iteration {count}: Entropy={new_entropy:.6f}, Diff={diff:.6f}, Coeffs={new_coeffs}")
        if count >= num_iterations:
            print(f"Reached maximum iterations ({num_iterations}). Stopping.")
            break
        if new_entropy < best_entropy:
            best_entropy = new_entropy
            best_coeffs = new_coeffs.copy()
            best_img = img.copy()

    return best_img, entropies, best_entropy, best_coeffs

def find_hotspots(img, n=10, std_coeff=2, neighborhood_size=7):
    threshold = img.mean() + std_coeff * img.std()
    local_max = img == maximum_filter(img, size=neighborhood_size)
    mask = local_max & (img > threshold)
    ys, xs = np.where(mask)
    vals = img[ys, xs]
    order = np.argsort(vals)[::-1][:n]
    xs_n = xs[order]
    ys_n = ys[order]
    vals_n = vals[order]
    peaks = np.column_stack((xs_n, ys_n, vals_n))
    return peaks

def extract_phase_error(phase_hist, peaks):
    all_dphase = []
    for i in range(len(peaks)):
        y = int(peaks[i, 0])
        x = int(peaks[i, 1])
        sig = phase_hist[y, x, :]
        amp = np.abs(sig)
        valid = amp > 0.2 * amp.max()
        dphase = np.angle(sig[1:] * np.conj(sig[:-1]))
        dphase = np.unwrap(dphase)
        valid_d = valid[1:] & valid[:-1]
        k = np.arange(len(dphase))
        good = valid_d & np.isfinite(dphase)
        if np.sum(good) < 10:
            continue
        trend = np.polyval(np.polyfit(k[good], dphase[good], deg=1), k)
        dphase = dphase - trend
        dphase -= np.mean(dphase[good])
        all_dphase.append(dphase)
    avg_dphase = np.median(np.array(all_dphase), axis=0)
    phase_error = np.concatenate([[0], np.cumsum(avg_dphase)])
    phase_error -= phase_error.mean()
    return phase_error

def phase_gradient_autofocus(
    positions,
    range_profiles,
    base_parameters,
    std_coeff=2,
    neighborhood_size=7,
    n_peaks=10,
    num_iterations=3
):
    # base_parameters["phase_sign"] = -1.0
    phase_error = np.zeros(range_profiles.shape[0])
    img = None

    for _ in range(num_iterations):
        img, c, d, phase_hist = bp(
            positions=positions,
            range_profiles=range_profiles,
            phase_error=phase_error,
            pga=True,
            **base_parameters
        )

        peaks = find_hotspots(
            img,
            n=n_peaks,
            std_coeff=std_coeff,
            neighborhood_size=neighborhood_size
        )

        phase_update = extract_phase_error(phase_hist, peaks)
        phase_update -= phase_update.mean()

        phase_error += phase_update
        phase_error -= phase_error.mean()

    focused, c, d, _ = bp(
        positions=positions,
        range_profiles=range_profiles,
        phase_error=phase_error,
        **base_parameters
    )

    return focused, img, phase_error


def parameterized_mea_animation(
    range_profiles,
    positions,
    base_parameters,
    save_path="parameterized_mea.mp4",
    stop_entropy_diff=5e-5,
    num_iterations=100,
    order=6,
    steps=[1e-3] * 4,
    learning_rates=[1e-3] * 4,
    poly_coeffs=[0] * 4,
    print_iterations_mod=10,
    coordinate_descent=False,
    dynamic_range_db=40,
    fps=8
):
    if order is None or steps is None or learning_rates is None:
        raise ValueError("order, steps, and learning_rates must be defined")

    entropies = []
    frames = []
    coeff_history = []

    diff = float("inf")
    new_coeffs = np.array(poly_coeffs).astype(np.float64)
    learning_rates = np.array(learning_rates).astype(np.float64)

    count = 0
    best_entropy = float("inf")
    best_coeffs = new_coeffs.copy()
    best_img = None

    while diff > stop_entropy_diff:
        if coordinate_descent:
            active_coeff = count % len(new_coeffs)
            coeff_mask = [1 if i == active_coeff else 0 for i in range(len(new_coeffs))]
        else:
            coeff_mask = None

        phase_error = taylor(poly_coeffs=new_coeffs, rp_len=positions.shape[0])
        img, c, d, _ = bp(
            positions=positions,
            range_profiles=range_profiles,
            phase_error=phase_error,
            **base_parameters
        )

        base_entropy = entropy_score(img)

        gradients = find_gradient_phase_error(
            range_profiles=range_profiles,
            positions=positions,
            base_parameters=base_parameters,
            coeff_mask=coeff_mask,
            poly_coeffs=new_coeffs,
            steps=steps,
            base_entropy=base_entropy
        )

        gradients = np.array(gradients, dtype=np.float64)
        new_coeffs -= gradients * learning_rates

        phase_error = taylor(poly_coeffs=new_coeffs, rp_len=positions.shape[0])
        img, c, d, _ = bp(
            positions=positions,
            range_profiles=range_profiles,
            phase_error=phase_error,
            **base_parameters
        )

        new_entropy = entropy_score(img)
        diff = abs(new_entropy - base_entropy)

        entropies.append(new_entropy)
        coeff_history.append(new_coeffs.copy())

        peak = np.max(np.abs(img))
        img_db = 20.0 * np.log10(np.abs(img) / (peak + 1e-12) + 1e-12)
        img_db = np.nan_to_num(img_db, nan=-120.0, posinf=0.0, neginf=-120.0)

        frames.append(img_db.copy())

        if new_entropy < best_entropy:
            best_entropy = new_entropy
            best_coeffs = new_coeffs.copy()
            best_img = img_db.copy()

        count += 1

        if count % print_iterations_mod == 0:
            print(f"Iteration {count}: Entropy={new_entropy:.6f}, Diff={diff:.6f}, Coeffs={new_coeffs}")

        if count >= num_iterations:
            print(f"Reached maximum iterations ({num_iterations}). Stopping.")
            break

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(frames[0], cmap="jet", origin="lower", vmax=peak, vmin=peak-20)
    title = ax.set_title(f"Iteration 1 | Entropy {entropies[0]:.6f}")
    fig.colorbar(im, ax=ax)

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"Iteration {i + 1} | Entropy {entropies[i]:.6f}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    if save_path.endswith(".gif"):
        anim.save(save_path, writer=PillowWriter(fps=fps))
    elif save_path.endswith(".mp4"):
        anim.save(save_path, writer=FFMpegWriter(fps=fps))
    else:
        raise ValueError("save_path must end with .gif or .mp4")

    plt.close(fig)

    return best_img, entropies, best_entropy, best_coeffs, coeff_history, save_path