clear; clc; close all;

%% --- 1. CONFIGURATION ---------------------------------------------------

% I/O
filename = ''; 

% GATING & FILTERING  
RANGE_GATE_FT       = 8.0;   % Silence signals closer than this (ft)
RANGE_GATE_FADE     = 1.0;   % Soft edge transition (ft)

% PROCESSING QUALITY
IMAGE_OVERSAMPLE    = 4;     % Higher = Smoother (4, 8, 16)
DB_DYNAMIC_RANGE    = 5;    % Contrast for final plot (Main Image)
MAX_GROUND_RANGE_FT = 20;    % Final Plot Limit (Y-Axis)
MAX_APERTURE_FT     = 20;    % Cross-Range Limit (X-Axis)

% DEBUGGING
DEBUG_RAW_SIGNALS     = true; 
DEBUG_SHOW_HYPERBOLAS = true; 
DEBUG_DYNAMIC_RANGE   = 20;  % Contrast for Debug Plot (dB)
DEBUG_MAX_BIN         = 500; % Vertical Limit for Debug Plot

% HARDWARE CONSTANTS
h_radar_ft          = 6;             
calibration_offset  = 15.5;  
c = 3E8; 
fstart = 2280E6; fstop = 2580E6; 
Tp = 5E-3;       
delta_x = 0.011;   
ir_thresh = 0.20; 

%% --- 2. DATA LOADING & SYNC ---------------------------------------------
if isempty(filename)
    [f, p] = uigetfile('*.wav', 'Select SAR Data');
    if isequal(f,0), error('No file selected.'); end
    filename = fullfile(p, f);
end

fprintf('Processing: %s\n', f);
[Y, FS] = audioread(filename);
radar_raw = Y(:, 2); 
sync_raw = Y(:, 3); 
ir_raw = Y(:, 4);
N = round(Tp * FS); 

% Find Triggers
sync_starts = find(diff(sync_raw > 0) == 1);
ir_edges = find(diff(abs(ir_raw) > ir_thresh) == 1); 
disp(length(ir_edges));
min_dist = FS * 0.025; 

valid_ir = []; last_c = -min_dist;
for k=1:length(ir_edges)
    if ir_edges(k) > last_c + min_dist
        valid_ir(end+1) = ir_edges(k); last_c = ir_edges(k); 
    end
end
num_profs = length(valid_ir);

%% --- 3. DEBUG: RAW SIGNAL CHECK -----------------------------------------
if DEBUG_RAW_SIGNALS
    figure('Name', 'Debug: Raw Signals', 'Color', 'w', 'NumberTitle', 'off');
    subplot(2,1,1);
    plot(ir_raw, 'Color', [0.7 0.7 0.7]); hold on;
    if ~isempty(valid_ir), plot(valid_ir, ir_raw(valid_ir), 'ro', 'LineWidth', 2); end
    title(sprintf('Detected %d Pulses', num_profs)); grid on; axis tight;
end

%% --- 4. PULSE EXTRACTION ------------------------------------------------
sif_time = zeros(num_profs, N);
for k=1:num_profs
    [~, idx_s] = min(abs(sync_starts - valid_ir(k))); 
    bst = round(sync_starts(idx_s));
    if bst + N - 1 <= length(radar_raw)
        sif_time(k, :) = radar_raw(bst : bst + N - 1).';
    end
end

% Remove DC Offset
sif_time = sif_time - mean(sif_time, 2);

%% --- 5. IMAGE RECONSTRUCTION (RMA) --------------------------------------
aperture_lim_m = MAX_APERTURE_FT * 0.3048;

[ProcessedImage, AxisCross, AxisDown, SlantCorrected] = runRMA_Script(...
    sif_time, delta_x, h_radar_ft, calibration_offset, aperture_lim_m, c, fstart, fstop, ...
    MAX_GROUND_RANGE_FT, DEBUG_SHOW_HYPERBOLAS, DEBUG_DYNAMIC_RANGE, DEBUG_MAX_BIN, ...
    IMAGE_OVERSAMPLE, RANGE_GATE_FT, RANGE_GATE_FADE);

%% --- 6. VISUALIZATION ---------------------------------------------------
img_db = 20*log10(abs(ProcessedImage));
range_gain = 30*log10(SlantCorrected * 0.3048);
img_db = img_db + range_gain.';

peak_val = max(img_db(:));%%-23.3; %%max(img_db(:))
clim_range = [peak_val - DB_DYNAMIC_RANGE, peak_val];

figure('Name', 'SAR Processor', 'Color', 'w', 'NumberTitle', 'off');
imagesc(AxisCross, AxisDown, img_db);
axis xy equal tight; colormap('turbo'); colorbar; grid on;
xlabel('Cross Range (ft)'); ylabel('Ground Range (ft)');
title({sprintf('File: %s', f), sprintf('Peak: %.1f dB', peak_val)}, 'Interpreter', 'none');

try clim(clim_range); catch, caxis(clim_range); end
fprintf('Done.\n');


% =========================================================================
% LOCAL FUNCTION: RMA ENGINE
% =========================================================================
function [S_img, AxisCross, AxisDown, SlantCorrected] = runRMA_Script(sif_time, delta_x, h_radar, cable_off, aperture_lim_m, c, fstart, fstop, max_range_ft, show_debug, debug_dyn_range, debug_max_bin, os_factor, gate_ft, gate_fade)
    
    % --- 1. RANGE COMPRESSION & GATING ---
    q = ifft(sif_time, [], 2); 
    
    if gate_ft > 0
        range_res_m = c / (2 * (fstop - fstart));
        dist_total_m = gate_ft * 0.3048; 
        
        stop_bins = round(dist_total_m / range_res_m);
        fade_bins = round((gate_fade * 0.3048) / range_res_m);
        
        [~, n_cols] = size(q);
        mask_row = ones(1, n_cols);
        ramp_up = linspace(0, 1, fade_bins);
        ramp_dn = linspace(1, 0, fade_bins);
        
        % Symmetric Gating (Left and Right wrap-around)
        if stop_bins < n_cols/2
            mask_row(1:stop_bins) = 0;
            mask_row(stop_bins+1 : stop_bins+fade_bins) = ramp_up;
            mask_row(end-stop_bins+1 : end) = 0;
            mask_row(end-stop_bins-fade_bins+1 : end-stop_bins) = ramp_dn;
        end
        q = q .* mask_row; 
    end

    % --- 2. DEBUG VIEW (HYPERBOLAS) ---
    if show_debug
        zpad_debug = 8192; 
        q_debug = ifft(sif_time, zpad_debug, 2); 
        
        % Apply scale-corrected mask to debug view
        if gate_ft > 0
             scale = zpad_debug / size(q,2);
             dbg_mask = ones(1, zpad_debug);
             stop_d = round(stop_bins * scale);
             dbg_mask(1:stop_d) = 0; 
             dbg_mask(end-stop_d+1:end) = 0;
             q_debug = q_debug .* dbg_mask;
        end
        
        figure('Name', 'Debug: Range Migration', 'Color', 'w', 'NumberTitle', 'off');
        d_img = 20*log10(abs(q_debug.'));
        imagesc(d_img); colormap('jet'); colorbar;
        ylim([0, debug_max_bin]); 
        d_max = max(d_img(:));
        % FIX: Using the debug_dyn_range passed from config
        try clim([d_max - debug_dyn_range, d_max]); catch, caxis([d_max - debug_dyn_range, d_max]); end
        title('Range Compressed History (Gated)'); xlabel('Pulse'); ylabel('Bin'); axis xy;
    end

    % --- 3. HILBERT & APERTURE ---
    hp = floor(size(q,2)/2); 
    sif = fft(q(:, hp+1:end), [], 2);
    
    if aperture_lim_m < size(sif,1)*delta_x
        sif = sif(1:round(aperture_lim_m/delta_x), :);
    end
    
    % --- 4. MATCHED FILTER (RMA) ---
    [nr, nc] = size(sif);
    sif = sif .* hanning(nc).'; % Windowing
    
    zpad = 2048; 
    szeros = zeros(zpad, nc); 
    idx_s = round((zpad-nr)/2);
    szeros(idx_s+1:idx_s+nr, :) = sif;
    S = fftshift(fft(szeros, [], 1), 1);
    
    Kx = linspace((-pi/delta_x), (pi/delta_x), zpad).';
    Rs = (10+6/12)*.3048; % Reference Range (~10.5 ft)
    fc = (fstop-fstart)/2 + fstart; 
    Kr = linspace(((4*pi/c)*(fc-(fstop-fstart)/2)), ((4*pi/c)*(fc+(fstop-fstart)/2)), nc);
    
    [Krg, Kxg] = meshgrid(Kr, Kx);
    S_mf = S .* exp(1j * Rs * sqrt(Krg.^2 - Kxg.^2));
    
    % --- 5. STOLT INTERPOLATION ---
    k_min = (4 * pi * fstart) / c;
    k_max = (4 * pi * fstop) / c;
    Ky_e = linspace(k_min, k_max, 2048); 
    [Ky_m, Kx_m] = meshgrid(Ky_e, Kx);
    Kr_q = sqrt(Ky_m.^2 + Kx_m.^2);
    
    S_st = interp2(Kr, Kx, S_mf, Kr_q, Kx_m, 'linear', 0);
    S_st(isnan(S_st))=1E-30;
    
    % --- 6. IMAGE FORMATION (OVERSAMPLED) ---
    ny = size(S_st,1) * os_factor; 
    nx = size(S_st,2) * os_factor;
    v = ifft2(S_st .* kaiser(size(S_st,2),0).', ny, nx) * (os_factor^2);
    S_img_full = rot90(v);
    
    % --- 7. MAPPING & CROP ---
    bw = c * (max(Ky_e) - min(Ky_e)) / (4 * pi);
    max_r_physical = (3E8*size(S_st,2)/(2*bw))*1/.3048; 
    
    px_per_m = size(S_img_full, 2) / (zpad*delta_x);
    hw_px = round((nr*delta_x)*px_per_m/2);
    mp = size(S_img_full,2)/2;
    
    c1 = max(1, floor(mp - hw_px)); 
    c2 = min(size(S_img_full,2), ceil(mp + hw_px));
    d1 = max(1, round(((cable_off)/max_r_physical)*size(S_img_full,1)));
    d2 = min(size(S_img_full,1), round(((max_range_ft + cable_off)/max_r_physical)*size(S_img_full,1)));
    
    S_img = S_img_full(d1:d2, c1:c2);
    
    % Axis Generation
    AxisCross = linspace(0, (nr * delta_x) / 0.3048, size(S_img,2));
    sl_raw = linspace(cable_off, max_range_ft + cable_off, size(S_img,1));
    
    % Slant to Ground Projection
    ground_cal = (real(sqrt(max(0, (sl_raw - cable_off).^2 - h_radar^2))) * 1.25) - 1.25;
    AxisDown = max(0, ground_cal);
    SlantCorrected = sqrt(AxisDown.^2 + h_radar^2);
end