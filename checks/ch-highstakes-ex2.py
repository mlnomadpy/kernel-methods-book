"""ch-highstakes-ex2: matched filtering as a noise-weighted inner product.

Detecting a known waveform h in noise is the cleanest possible kernel story: the
optimal statistic is the noise-weighted inner product <d, h>, and detection is the
projection of the data onto the template. In white noise the inner product is the
plain dot product; the optimal SNR is ||h||/sigma. We set that to 8, inject the
template, slide it, and read the peak.
"""
import numpy as np

rng = np.random.default_rng(11)

N = 512
t = np.linspace(0, 1, N)
# a Newtonian-style chirp: rising frequency, tapered amplitude
env = np.exp(-((t - 0.55) ** 2) / (2 * 0.12 ** 2))
phase = 2 * np.pi * (8 * t + 20 * t ** 2)
h = env * np.sin(phase)
h = h - h.mean()
hnorm = np.linalg.norm(h)

snr_opt = 8.0
sigma = hnorm / snr_opt
t0_true = 60                                    # true arrival offset (samples)
data = sigma * rng.standard_normal(3 * N)       # a padded noise stream
data[N + t0_true:N + t0_true + N] += h          # inject the template at a known offset

# matched filter: slide the template over lags whose window fits the buffer.
# the normalized statistic rho(L) = <d, h>/(sigma ||h||) has unit-variance noise,
# so at the true lag it equals the optimal SNR plus a standard normal.
lags = np.arange(-N // 2, N // 2)
rho = np.array([np.dot(data[N + L:N + L + N], h) / (sigma * hnorm) for L in lags])
peak_idx = lags[np.argmax(rho)]
peak_val = rho.max()
# loudest pure-noise excursion, away from the signal
mask = np.abs(lags - t0_true) > N // 4
noise_max = rho[mask].max()

# Negative control: a chirp with the wrong frequency evolution.
phase_bad = 2 * np.pi * (15 * t + 4 * t ** 2)
h_bad = env * np.sin(phase_bad)
h_bad -= h_bad.mean()
rho_bad = np.array([
    np.dot(data[N + L:N + L + N], h_bad) /
    (sigma * np.linalg.norm(h_bad))
    for L in lags
])
bad_at_truth = rho_bad[lags == t0_true][0]
print("matched filter as a noise-weighted inner product")
print(f"  optimal SNR rho = ||h||/sigma : {snr_opt:.1f}")
print(f"  template norm ||h||           : {hnorm:.3f}   sigma = {sigma:.4f}")
print(f"  peak statistic                : {peak_val:.2f}  at lag {peak_idx} (true {t0_true})")
print(f"  loudest pure-noise excursion  : {noise_max:.2f}")
print(f"  mismatched statistic at truth : {bad_at_truth:.2f}")

assert peak_idx == t0_true
assert abs(hnorm - 7.372) < 5e-4
assert abs(sigma - 0.9215) < 5e-5
assert peak_val > 7.0
assert bad_at_truth < 2.0
assert np.isfinite(rho).all() and np.isfinite(rho_bad).all()
