"""Which side a sound came from, using a stereo microphone pair (e.g. a chest-worn Insta360's two mics).

Cross-correlates the two channels (GCC-PHAT) to find the arrival-time difference, then converts it to an angle
with the known mic spacing. Only a stereo pair sharing one clock works; two separate devices cannot be aligned.
Self-check: python direction.py --selftest
"""
import sys

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s
CENTER_DEG = 20  # inside this cone the sound counts as "ahead"


def delay_seconds(left, right, rate):
    """Arrival-time difference right - left (positive = reached the right mic first) via GCC-PHAT."""
    n = len(left) + len(right)
    spec = np.fft.rfft(left, n) * np.conj(np.fft.rfft(right, n))
    spec /= np.abs(spec) + 1e-12
    corr = np.fft.irfft(spec, n)
    max_shift = n // 2
    corr = np.concatenate((corr[-max_shift:], corr[: max_shift + 1]))
    shift = int(np.argmax(corr)) - max_shift
    return shift / rate


def direction(stereo, rate, spacing_m):
    """Returns (side, angle_deg): side is 'left', 'center' or 'right'; angle is -90..90 (negative = left)."""
    left, right = stereo[:, 0].astype(np.float64), stereo[:, 1].astype(np.float64)
    tau = delay_seconds(left, right, rate)
    sin_angle = max(-1.0, min(1.0, tau * SPEED_OF_SOUND / spacing_m))
    angle = float(np.degrees(np.arcsin(sin_angle)))
    side = "center" if abs(angle) <= CENTER_DEG else ("right" if angle > 0 else "left")
    return side, round(angle, 1)


def selftest():
    rate, spacing = 16000, 0.08
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(rate // 2)
    # A source on the right reaches the right mic first: delay the LEFT channel by the travel time across the pair.
    lag = int(round(spacing / SPEED_OF_SOUND * rate * 0.7))  # 0.7 -> ~44 degrees
    stereo = np.stack([np.concatenate([np.zeros(lag), noise[:-lag]]), noise], axis=1)
    side, angle = direction(stereo, rate, spacing)
    assert side == "right" and 30 < angle < 60, (side, angle)
    stereo = np.stack([noise, np.concatenate([np.zeros(lag), noise[:-lag]])], axis=1)
    side, angle = direction(stereo, rate, spacing)
    assert side == "left" and -60 < angle < -30, (side, angle)
    side, angle = direction(np.stack([noise, noise], axis=1), rate, spacing)
    assert side == "center" and abs(angle) < 5, (side, angle)
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
