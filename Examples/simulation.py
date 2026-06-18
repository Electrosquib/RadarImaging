import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

c = 3e8
fstart = 2.7e9
fstop = 3.0e9
fc = (fstart + fstop) / 2
B = fstop - fstart
num_freq = 512
dr = c / (2 * B)

freqs = np.linspace(-B / 2, B / 2, num_freq)

scatterers = np.array([
    [-2.8, 4.0, 0.0, 1.0],
    [2.5, 5.0, 0.0, 0.8],
    [1.0, 1.0, 0.0, 0.7],
    [-1.2, 6.5, 0.0, 0.6]
])

aperture_width = 4.0
lam = c / fc
dx_max = lam / 4
num_positions = int(np.ceil(aperture_width / dx_max)) + 1
radar_height = 1.8
imsize = (4.0, 4.0)

radar_x = np.linspace(-aperture_width / 2, aperture_width / 2, num_positions)

coords = np.zeros((num_positions, 3))
coords[:,0] = radar_x

radar_positions = np.column_stack([
    radar_x,
    np.zeros(num_positions),
    np.full(num_positions, radar_height)
])

def make_range_profile(pos):
    xyz = scatterers[:, :3]
    amp = scatterers[:, 3]
    R = np.linalg.norm(xyz - pos, axis=1)
    carrier = np.exp(-1j * 4 * np.pi * fc * R / c)
    phase = np.exp(-1j * 4 * np.pi * freqs[None, :] * R[:, None] / c)
    S = np.sum((amp * carrier)[:, None] * phase, axis=0)
    rp = np.fft.ifft(S)
    return rp

range_profiles = np.array([make_range_profile(p) for p in radar_positions])
ranges = np.arange(num_freq) * dr

fig, (ax_scene, ax_rp, ax_hist, ax_bp) = plt.subplots(1, 4, figsize=(15, 4))

ax_scene.set_xlim(-3, 3)
ax_scene.set_ylim(0, 8)
ax_scene.set_aspect("equal")
ax_scene.set_xlabel("Cross range (m)")
ax_scene.set_ylabel("Downrange (m)")
ax_scene.set_title("SAR Scene")

ax_scene.scatter(scatterers[:, 0], scatterers[:, 1], s=80 * scatterers[:, 3])
path_line, = ax_scene.plot([], [], linewidth=2)
radar_dot, = ax_scene.plot([], [], "ro")
look_lines = [ax_scene.plot([], [], "r-", alpha=0.25)[0] for _ in scatterers]

ax_scene.set_xlim(-3, 3)
ax_scene.set_ylim(0, 8)
ax_bp.set_xlabel("Crossrange (m)")
ax_bp.set_ylabel("Downrange (m)")
ax_bp.set_title("Current Backprojection")
bp_img0 = np.zeros((50, 50))
bp_artist = ax_bp.imshow(
    bp_img0,
    cmap="jet",
    origin="lower",
    extent=(-imsize[0] / 2, imsize[0] / 2, 0, 8),
    vmin=-60,
    vmax=0,
    animated=True
)
plt.colorbar(bp_artist, ax=ax_bp)

ax_rp.set_xlim(0, 10)
ax_rp.set_ylim(-80, 0)
ax_rp.set_xlabel("Range (m)")
ax_rp.set_ylabel("Magnitude/ (dB)")
ax_rp.set_title("Current Range Profile")
rp_line, = ax_rp.plot([], [])
rp_mag = np.abs(range_profiles)
rp_db_all = 20 * np.log10(rp_mag / np.max(rp_mag) + 1e-12)

hist_artist = ax_hist.imshow(
    rp_db_all.T,
    cmap="jet",
    aspect="auto",
    origin="lower",
    extent=[0, num_positions - 1, ranges[0], ranges[-1]],
    vmin=-60,
    vmax=0,
    animated=True
)

ax_hist.set_ylim(0, 10)
ax_hist.set_xlabel("Slow time / pulse")
ax_hist.set_ylabel("Fast time / range (m)")
ax_hist.set_title("Range Profile History")

hist_line = ax_hist.axvline(0, color="white", linewidth=1.5)
plt.colorbar(hist_artist, ax=ax_hist)

def interp_rp(k, range_profile):
    if k < 0 or k >= len(range_profile) - 1:
        return 0j
    x1 = int(np.floor(k))
    x2 = x1 + 1
    a = k - x1
    return (1 - a) * range_profile[x1] + a * range_profile[x2]

def bp(i, resolution=50):
    x_grid = np.linspace(-imsize[0] / 2, imsize[0] / 2, resolution)
    y_grid = np.linspace(0, 8, resolution)

    img = np.zeros((resolution, resolution), np.complex128)
    range_resolution_m = c / (2 * B)

    used_positions = radar_positions[:i + 1]
    used_profiles = range_profiles[:i + 1]

    for yi, y in enumerate(y_grid):
        for xi, x in enumerate(x_grid):
            px_loc = np.array([x, y, 0.0])

            for j, pos in enumerate(used_positions):
                d = np.linalg.norm(px_loc - pos)
                k = d / range_resolution_m
                s = interp_rp(k, used_profiles[j])
                img[yi, xi] += s * np.exp(1j * 4 * np.pi * fc * d / c)

    img = np.abs(img)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if np.max(img) > 0:
        img = 20 * np.log10(img / np.max(img) + 1e-12)
    else:
        img = np.full_like(img, -80.0)

    return img

def update(i):
    pos = radar_positions[i]
    rp = range_profiles[i]
    bp_img = bp(i)
    bp_artist.set_array(bp_img)
    rp_db = 20 * np.log10(np.abs(rp) / np.max(np.abs(rp)) + 1e-12)
    radar_dot.set_data([pos[0]], [pos[1]])
    path_line.set_data(radar_positions[:i + 1, 0], radar_positions[:i + 1, 1])
    hist_line.set_xdata([i, i])
    for line, sc in zip(look_lines, scatterers):
        line.set_data([pos[0], sc[0]], [pos[1], sc[1]])
    rp_line.set_data(ranges, rp_db)
    return [radar_dot, path_line, rp_line, hist_line, bp_artist, *look_lines]

ani = FuncAnimation(fig, update, frames=range(1, num_positions), interval=40, blit=True)
ani.save("sar_animation.mp4", writer="ffmpeg", fps=25, dpi=150)
plt.show()