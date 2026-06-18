"""
RMA SAR Processor — Python port of the MATLAB script. 

Dependencies:
    pip install numpy scipy matplotlib soundfile

Usage:
    python sar_processor.py                 # opens a file picker
    python sar_processor.py path/to/file.wav

Notes on the port:
  - MATLAB is 1-based; Python is 0-based. Index math has been adjusted.
  - MATLAB `ifft(X, [], 2)` -> np.fft.ifft(X, axis=1).
  - MATLAB `hanning(n)` is the symmetric Hann *without* the zero endpoints
    (i.e. scipy.signal.windows.hann(n, sym=True) reproduces MATLAB hann; but
    MATLAB's legacy `hanning` == hann interior). We use the periodic-free
    symmetric Hann, which matches MATLAB `hanning`.
  - MATLAB `kaiser(n, beta)` with beta=0 is a rectangular window (all ones);
    we replicate that directly.
  - MATLAB `interp2` with 'linear' + 0 fill -> scipy RegularGridInterpolator.
"""

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

class MITRadar:
    def __init__(self, filename):
        """
        filename: path to .WAV file with the following channels:
            CH1: Unused
            CH2: Radar / SIF signal (Real-only baseband dechirped samples)
            CH3: Sync pulse
            CH4: IR encoder / position trigger

        # GATING & FILTERING
        self.RANGE_GATE_FT       = 8.0    # Silence signals closer than this (ft)
        self.RANGE_GATE_FADE     = 1.0    # Soft edge transition (ft)
        self.CLUTTER_REMOVAL     = True   # Subtract slow-time mean to kill stationary clutter/coupling
                                    # (the flat horizontal bands RMA rejected for free)

        # PROCESSING QUALITY
        self.IMAGE_OVERSAMPLE    = 1       # Range-profile oversampling (finer range bins). 4/8/16
        self.BP_IMAGE_PX         = 400     # Back-projection output grid size (NxN pixels)
        self.DB_DYNAMIC_RANGE    = 5       # Contrast for final plot (Main Image)
        self.MAX_GROUND_RANGE_FT = 20      # Final Plot Limit (Y-Axis)
        self.MAX_APERTURE_FT     = 20      # Cross-Range Limit (X-Axis, full width)

        # POST-PROCESSING
        self.APERTURE_WINDOW     = True   # Hann taper across the aperture (slow-time) to
                                    # suppress cross-range sidelobe streaks. Safe win;
                                    # costs a little cross-range resolution.

        # DEBUGGING
        self.DEBUG_SHOW_HYPERBOLAS = False  # Plot the range-compressed history (hyperbolas) after gating.
        self.DEBUG_RAW_SIGNALS     = False # Plot the raw IR signal with detected pulse starts (red circles).
        self.DEBUG_DYNAMIC_RANGE   = 20    # Contrast for Debug Plot (dB)
        self.DEBUG_MAX_BIN         = 500   # Vertical Limit for Debug Plot
        self.DEBUG_PER_PULSE       = False # Dump a "how back-projection builds up" figure:
                                    # individual per-pulse arcs + cumulative sums.
                                    # Diagnostic only; saves its own PNG. Off by default.
        self.COMPARISON_VIEW       = False # Dump a two-panel "Raw Backprojection" vs
                                    # "Processed BP" figure (jet linear-magnitude raw,
                                    # no clutter removal/threshold, next to the clean
                                    # dB version). Saves its own PNG. Off by default.

        # HARDWARE CONSTANTS
        self.H_RADAR_FT          = 6
        self.CALIBRATION_OFFSET  = 15.5
        self.C = 3e8
        self.FSTART = 2280e6
        self.FSTOP  = 2580e6
        self.TP = 5e-3
        self.DELTA_X = 0.011
        self.IR_THRESH = 0.20

        self.FT = 0.3048  # meters per foot
        """
        self.filename = filename

        # GATING & FILTERING
        self.RANGE_GATE_FT       = 8.0    # Silence signals closer than this (ft)
        self.RANGE_GATE_FADE     = 1.0    # Soft edge transition (ft)
        self.CLUTTER_REMOVAL     = True   # Subtract slow-time mean to kill stationary clutter/coupling
                                    # (the flat horizontal bands RMA rejected for free)

        # PROCESSING QUALITY
        self.IMAGE_OVERSAMPLE    = 1       # Range-profile oversampling (finer range bins). 4/8/16
        self.BP_IMAGE_PX         = 400     # Back-projection output grid size (NxN pixels)
        self.DB_DYNAMIC_RANGE    = 5       # Contrast for final plot (Main Image)
        self.MAX_GROUND_RANGE_FT = 20      # Final Plot Limit (Y-Axis)
        self.MAX_APERTURE_FT     = 20      # Cross-Range Limit (X-Axis, full width)

        # POST-PROCESSING
        self.APERTURE_WINDOW     = True   # Hann taper across the aperture (slow-time) to
                                    # suppress cross-range sidelobe streaks. Safe win;
                                    # costs a little cross-range resolution.

        # DEBUGGING
        self.DEBUG_SHOW_HYPERBOLAS = False  # Plot the range-compressed history (hyperbolas) after gating.
        self.DEBUG_RAW_SIGNALS     = False # Plot the raw IR signal with detected pulse starts (red circles).
        self.DEBUG_DYNAMIC_RANGE   = 20    # Contrast for Debug Plot (dB)
        self.DEBUG_MAX_BIN         = 500   # Vertical Limit for Debug Plot
        self.DEBUG_PER_PULSE       = False # Dump a "how back-projection builds up" figure:
                                    # individual per-pulse arcs + cumulative sums.
                                    # Diagnostic only; saves its own PNG. Off by default.
        self.COMPARISON_VIEW       = False # Dump a two-panel "Raw Backprojection" vs
                                    # "Processed BP" figure (jet linear-magnitude raw,
                                    # no clutter removal/threshold, next to the clean
                                    # dB version). Saves its own PNG. Off by default.

        # HARDWARE CONSTANTS
        self.H_RADAR_FT          = 6
        self.CALIBRATION_OFFSET  = 15.5
        self.C = 3e8
        self.FSTART = 2280e6
        self.FSTOP  = 2580e6
        self.TP = 5e-3
        self.DELTA_X = 0.011
        self.IR_THRESH = 0.20

        self.FT = 0.3048  # meters per foot

    def rp(self, sif_time, delta_x, h_radar, cable_off, c, fstart, fstop,
                        max_range_ft, max_aperture_ft, show_debug, debug_dyn_range,
                        debug_max_bin, os_factor, gate_ft, gate_fade,
                        clutter_removal=True, aperture_window=True):
        """Time-domain back-projection SAR reconstruction.

        For every antenna position, the matched range profile is computed, then each
        image pixel accumulates the (phase-compensated) complex sample at the exact
        round-trip slant range from that antenna position to the pixel. Unlike RMA,
        there is no Stolt interpolation and no uniform-aperture assumption, so it
        degrades gracefully on uneven pulse spacing and odd geometries — which is
        usually why a back-projection image looks more stable than an RMA one.

        Returns: S_img (complex), AxisCross (ft), AxisDown (ft), SlantCorrected (ft)
        """
        num, N = sif_time.shape
        ft = 0.3048
        bw = fstop - fstart
        fc = (fstart + fstop) / 2

        # --- 0. SLOW-TIME CLUTTER REMOVAL ---
        # Subtract the across-pulse (slow-time) mean. Anything that doesn't change as
        # the antenna moves -- direct antenna coupling, cable reflections, stationary
        # background -- is removed. These show up as flat horizontal bands in the
        # range-compressed history and, if left in, dominate the coherent sum. RMA
        # suppresses them implicitly; back-projection needs this done explicitly.
        if clutter_removal:
            sif_time = sif_time - sif_time.mean(axis=0, keepdims=True)

        # --- 1. RANGE COMPRESSION (windowed, zero-padded IFFT -> fine range bins) ---
        win = np.hanning(N)[np.newaxis, :]
        n_fft = max(os_factor, 4) * N
        rp = np.fft.ifft(sif_time * win, n=n_fft, axis=1)   # complex range profiles
        dr = c / (2 * bw) * (N / n_fft)                      # slant-range bin spacing (m)
        slant_axis = np.arange(n_fft) * dr                   # slant range from antenna (m)

        # keep positive ranges within a sane maximum
        max_slant_m = (max_range_ft + cable_off + 5) * ft
        keep = slant_axis <= max_slant_m
        slant_axis = slant_axis[keep]
        rp = rp[:, keep]

        # --- 2. RANGE GATE (null near-range cable / direct coupling) ---
        if gate_ft > 0:
            gate_m = gate_ft * ft
            fade_m = max(gate_fade, 1e-6) * ft
            ramp = np.clip((slant_axis - gate_m) / fade_m, 0.0, 1.0)  # 0 below gate -> 1 after fade
            rp = rp * ramp[np.newaxis, :]

        # --- 3. ANTENNA POSITIONS (uniform spacing, centered) ---
        x_ant = np.arange(num) * delta_x
        if max_aperture_ft > 0:
            # limit aperture if requested (keep central portion)
            ap_lim_m = max_aperture_ft * ft
            if x_ant[-1] > ap_lim_m:
                mid = x_ant.mean()
                sel = np.abs(x_ant - mid) <= ap_lim_m / 2
                x_ant = x_ant[sel]
                rp = rp[sel, :]
                num = rp.shape[0]
        x_ant = x_ant - x_ant.mean()

        # --- 3b. APERTURE WINDOW (slow-time Hann taper) ---
        # The aperture is a finite, hard-edged set of pulses; that rectangular
        # extent produces strong cross-range sidelobes (the vertical streaks fanning
        # off each target). Tapering the per-pulse amplitudes with a Hann window
        # across the aperture suppresses those sidelobes, at the cost of a little
        # cross-range resolution (the mainlobe widens slightly).
        if aperture_window and num > 2:
            rp = rp * np.hanning(num)[:, np.newaxis]

        # --- DEBUG: range-compressed history (hyperbolas) ---
        if show_debug:
            plt.figure('Debug: Range Migration', facecolor='w')
            d_img = 20 * np.log10(np.abs(rp.T) + 1e-30)
            plt.imshow(d_img, aspect='auto', cmap='jet', origin='lower')
            plt.colorbar()
            plt.ylim([0, min(debug_max_bin, rp.shape[1])])
            d_max = np.max(d_img)
            plt.clim([d_max - debug_dyn_range, d_max])
            plt.title('Range Compressed History (Gated)')
            plt.xlabel('Pulse'); plt.ylabel('Range bin')

        positions = np.array([x_ant, np.zeros_like(x_ant), np.full_like(x_ant, h_radar * 0.3048)]).T
        return rp, positions, dr

    def compute_range_profiles(self):
        print(f'Processing: {self.filename}')
        Y, FS = sf.read(self.filename)            # Y shape: (samples, channels)

        # MATLAB columns 2,3,4 -> Python 0-based 1,2,3
        radar_raw = Y[:, 1]
        sync_raw  = Y[:, 2]
        ir_raw    = Y[:, 3]
        N = round(self.TP * FS)

        # --- Find triggers ---
        # diff(sync_raw > 0) == 1  -> rising edges. np.diff drops one sample, so the
        # MATLAB find() index (1-based) of a rising edge maps to the same array index
        # here once we account for 0-based: edge at i means sample i+1 went high.
        sync_bool = (sync_raw > 0).astype(int)
        sync_starts = np.where(np.diff(sync_bool) == 1)[0] + 1

        ir_bool = (np.abs(ir_raw) > self.IR_THRESH).astype(int)
        ir_edges = np.where(np.diff(ir_bool) == 1)[0] + 1

        min_dist = FS * 0.025
        valid_ir = []
        last_c = -min_dist
        for k in range(len(ir_edges)):
            if ir_edges[k] > last_c + min_dist:
                valid_ir.append(ir_edges[k])
                last_c = ir_edges[k]
        valid_ir = np.array(valid_ir, dtype=int)
        num_profs = len(valid_ir)

        # --- DEBUG: raw signal check ---
        if self.DEBUG_RAW_SIGNALS:
            plt.figure('Debug: Raw Signals', facecolor='w')
            plt.subplot(2, 1, 1)
            plt.plot(ir_raw, color=[0.7, 0.7, 0.7])
            if valid_ir.size:
                plt.plot(valid_ir, ir_raw[valid_ir], 'ro', linewidth=2)
            plt.title(f'Detected {num_profs} Pulses')
            plt.grid(True)
            plt.axis('tight')

        # --- Pulse extraction ---
        sif_time = np.zeros((num_profs, N))
        for k in range(num_profs):
            idx_s = np.argmin(np.abs(sync_starts - valid_ir[k]))
            bst = int(round(sync_starts[idx_s]))
            if bst + N <= len(radar_raw):
                sif_time[k, :] = radar_raw[bst:bst + N]

        # Remove DC offset (per-row mean, like MATLAB mean(...,2))
        sif_time = sif_time - sif_time.mean(axis=1, keepdims=True)

        rp, positions, dr = self.rp(
            sif_time, 
            self.DELTA_X, 
            self.H_RADAR_FT, 
            self.CALIBRATION_OFFSET,
            self.C, 
            self.FSTART, 
            self.FSTOP, 
            self.MAX_GROUND_RANGE_FT, 
            self.MAX_APERTURE_FT,
            self.DEBUG_SHOW_HYPERBOLAS, 
            self.DEBUG_DYNAMIC_RANGE, 
            self.DEBUG_MAX_BIN,
            self.IMAGE_OVERSAMPLE, 
            self.RANGE_GATE_FT, 
            self.RANGE_GATE_FADE,
            clutter_removal=self.CLUTTER_REMOVAL,
            aperture_window=self.APERTURE_WINDOW)
        return rp, positions, dr