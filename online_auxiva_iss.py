#!/usr/bin/env python3
"""online_auxiva_iss.py — Online AuxIVA with Iterative Source Steering.

Standalone implementation of Algorithm 1 from

    T. Nakashima and N. Ono, "Inverse-free Online Independent Vector Analysis
    with Flexible Iterative Source Steering", arXiv:2209.00937 (2022).

OnlineAuxIVA_ISS itself depends only on NumPy, and shares no code with the IP
baseline module, so either can be used on its own. _demo() additionally uses
pyroomacoustics, only to generate room impulse responses for the mixing —
see its docstring.

--------------------------------------------------------------------------
What this implements
--------------------------------------------------------------------------

Per time frame t, for N_iter sweeps:

  1. Weighted covariance, autoregressive (Eq. 9), for every source k:

         U_kft <- alpha * U_kf(t-1) + (1 - alpha) * phi(r_kt) x_ft x_ft^H

     with r_kt = sqrt(sum_f |w_kf^H x_ft|^2)   (Eq. 5)
     and phi(r) = 1/(2r) for the spherical Laplace source model, or
         phi(r) = F/r^2  for the time-varying Gaussian model.

  2. ISS update of the demixing matrix (Eqs. 10-11), for k in I_k:

         W_f <- W_f - v_kf w_kf^H

         v_mkf = (w_mf^H U_mf w_kf) / (w_kf^H U_mf w_kf)     for m != k
         v_kkf = 1 - (w_kf^H U_kf w_kf)^(-1/2)               for m == k

  No matrix inversion appears anywhere in the update.

--------------------------------------------------------------------------
Two points that are easy to get wrong
--------------------------------------------------------------------------

* The recursion in step 1 is anchored to U_kf(t-1), the PREVIOUS frame's
  covariance — not to the current sweep's iterate. Recomputing r_kt each
  sweep is the point of the inner loop; decaying by alpha again is not.
  Applying the recursion to the running iterate would give an effective
  forgetting factor of alpha^N_iter and inject x x^H N_iter times per frame.
  `step()` therefore snapshots U_{t-1} on entry and rebuilds from it.

* This uses the covariance form of the ISS update (Eq. 11), not the cheaper
  batch form (Eqs. 12-13) that works directly on the separated outputs y.
  The batch form is unusable online: separating the incoming frame requires
  W_ft explicitly, and Eqs. 12-13 never form it. So "inverse-free" here buys
  O(K^2) quadratic forms in place of an O(K^3) solve — a real but modest
  saving that only becomes visible at larger K.

--------------------------------------------------------------------------
Flexible update (Sec. IV-B)
--------------------------------------------------------------------------

The ISS update of Eq. 10 is equivalent to updating the *steering vector*
a_kf, i.e. a COLUMN of the mixing matrix (Eq. 14). When only some sources
move, only those columns of A go stale, so the update can be restricted to
I_k = {indices of the moving sources}. Pass `active` to `step()` to do this.
IP has no equivalent: it updates a ROW of W, and no row-wise operation fixes
a single column of A, which is why a partial IP update cannot track.

The paper assumes the moving source is known. Detecting it online — deciding
which separated output's steering vector has drifted, under reverberation —
is left as future work there, and is not attempted here.

--------------------------------------------------------------------------
_demo(): 2 mics / 2 real sources — what to expect, and why
--------------------------------------------------------------------------

_demo() runs the determined K = 2 case (2 mics = 2 sources) on two real
speech recordings, mixed through a simulated room (pyroomacoustics RIRs,
RT60 = 0.2 s) plus additive noise (see its docstring). Getting it to run
cleanly, and produce numbers that meant what they appeared to, surfaced four
real issues worth knowing about if you change the setup:

* The array's spatial-aliasing limit c / 2d is a real conditioning cliff:
  above it, different directions alias to the same inter-mic phase, and
  condition numbers in the hundreds to thousands were measured there on
  real speech (this was diagnosed against the anechoic steering-vector
  model this demo used before adding real RIRs, but it's a property of the
  array geometry, present with or without reverberation). _demo() restricts
  separation to [ACTIVE_MIN_HZ, ACTIVE_MAX_HZ] and zeros everything else,
  rather than passing the raw mixture through — pass-through looks fuller
  on playback, but it's identical unseparated content in both output
  channels, and it inflates correlation/SDR against either source without
  reflecting anything the algorithm did.
* Real speech has genuine silences that a synthetic bursty envelope
  doesn't. phi(r) = 1/(2r) blows up as r -> 0, and one bad frame's update
  can persist since the recursion has no mechanism to undo it — hence
  the energy gate in _demo() (skip adapting, not skip separating, on
  near-silent frames).
* The NumPy-only ISTFT's overlap-add normalization divides by the summed
  squared window, which a Hann window tapers to exactly 0 at the very
  edges of the signal — not quite 0 a sample or two in, but small enough
  (~1e-9) that ordinary floating-point residue in the numerator becomes a
  single huge spurious spike there. That one sample being the loudest in
  the file was silently defeating peak-normalization on write-out (the
  real audio got crushed ~100x quieter) and, separately, dominating the
  noise-energy term in SI-SDR enough to make a source that was actually
  separating reasonably well score as a total failure. _istft() clamps the
  overlap-add floor to half the steady-state level instead of an absolute
  epsilon, which fixed both. If a metric or an output file from this kind
  of NumPy-only OLA implementation looks implausible, check the first/last
  few samples before suspecting the algorithm.
* Reverberation this long relative to the STFT window (RT60 = 0.2 s vs. a
  1024-sample/64 ms window) makes AuxIVA-ISS's narrowband (per-bin
  instantaneous mixing) assumption considerably less accurate than in the
  anechoic case, so separation is measurably harder here — expected, and
  in line with what the algorithm's own paper reports at a similar RT60
  (Sec. V), not a new bug. Evaluation scores each output against source
  k's own reverberant image at the reference mic, not the dry source:
  scoring against the dry source would penalize the separator for
  reverberation it was never trying to remove.
"""

from __future__ import annotations

import wave
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pyroomacoustics as pra

EPS = 1.0e-10


class OnlineAuxIVA_ISS:
    """Online AuxIVA with Iterative Source Steering.

    Parameters
    ----------
    n_src : int
        Number of sources K. Determined case only: equals the number of mics.
    n_freq : int
        Number of frequency bins F.
    alpha : float
        Forgetting factor in [0, 1) for the covariance recursion (Eq. 9).
    n_iter : int
        Sweeps per frame (the paper uses 2).
    model : {"laplace", "gauss"}
        Source model, selecting phi(r).
    w_init, u_init : ndarray, optional
        Initial W_f0 (F, K, K) and U_kf0 (K, F, K, K). Default: identity
        demixing matrices and 0.001 * I covariances, as in the paper.

    Attributes
    ----------
    W : ndarray, shape (F, K, K)
        Current demixing matrices. Row k holds w_kf^H.
    U : ndarray, shape (K, F, K, K)
        Current weighted covariances.
    """

    def __init__(
        self,
        n_src: int,
        n_freq: int,
        alpha: float = 0.99,
        n_iter: int = 2,
        model: str = "laplace",
        w_init: np.ndarray | None = None,
        u_init: np.ndarray | None = None,
    ):
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1): {alpha}")
        if model not in ("laplace", "gauss"):
            raise ValueError(f"model must be 'laplace' or 'gauss': {model}")

        self.K = int(n_src)
        self.F = int(n_freq)
        self.alpha = float(alpha)
        self.n_iter = int(n_iter)
        self.model = model

        if w_init is None:
            self.W = np.broadcast_to(
                np.eye(self.K, dtype=complex), (self.F, self.K, self.K)
            ).copy()
        else:
            self.W = np.array(w_init, dtype=complex).reshape(self.F, self.K, self.K)

        if u_init is None:
            self.U = np.broadcast_to(
                0.001 * np.eye(self.K, dtype=complex),
                (self.K, self.F, self.K, self.K),
            ).copy()
        else:
            self.U = np.array(u_init, dtype=complex).reshape(
                self.K, self.F, self.K, self.K
            )

    # ---- source model -----------------------------------------------------
    def _phi(self, r: np.ndarray) -> np.ndarray:
        """Weighting function phi(r) for the chosen source model."""
        if self.model == "laplace":
            return 1.0 / (2.0 * (r + EPS))
        return self.F / np.maximum(r**2, EPS)

    # ---- phase 1: covariance recursion (Eqs. 5, 9) ------------------------
    def _update_covariances(self, u_prev: np.ndarray, outer: np.ndarray, X: np.ndarray):
        """Rebuild U from u_prev (= U_{t-1}) using this sweep's r_kt.

        Runs for ALL k regardless of which sources are being steered: the
        covariances are the algorithm's memory of the scene, and letting them
        go stale for the frozen sources would corrupt the v_mkf denominators.
        """
        u_new = np.empty_like(u_prev)
        for k in range(self.K):
            y_k = np.einsum("fi,fi->f", self.W[:, k, :], X)     # y_kft, all f
            r_k = np.sqrt(np.sum(np.abs(y_k) ** 2))             # r_kt, Eq. 5
            phi = self._phi(r_k)
            u_new[k] = self.alpha * u_prev[k] + (1.0 - self.alpha) * phi * outer
        self.U = u_new

    # ---- phase 2: ISS update (Eqs. 10-11) ---------------------------------
    def _update_demixing(self, active: Iterable[int]):
        for k in active:
            # Snapshot w_kf^H before the m-loop: every rank-1 update in this
            # k-step uses the PRE-update row k, matching the simultaneous
            # form of Eq. 10. (Each v_mkf reads W[:, m, :], which has not
            # been written yet at that point in the loop, so the sequential
            # in-place writes are equivalent to the simultaneous update.)
            row_k = self.W[:, k, :].copy()                  # w_kf^H : (F, K)
            w_k = np.conj(row_k)                             # w_kf

            Uk_wk = np.einsum("fij,fj->fi", self.U[k], w_k)
            denom_kk = np.einsum("fi,fi->f", row_k, Uk_wk).real
            v_kk = 1.0 - 1.0 / np.sqrt(np.maximum(denom_kk, EPS))

            for m in range(self.K):
                if m == k:
                    v_m = v_kk
                else:
                    Um_wk = np.einsum("fij,fj->fi", self.U[m], w_k)
                    num = np.einsum("fi,fi->f", self.W[:, m, :], Um_wk)   # w_m^H U_m w_k
                    den = np.einsum("fi,fi->f", row_k, Um_wk).real         # w_k^H U_m w_k
                    v_m = num / np.maximum(den, EPS)
                self.W[:, m, :] -= v_m[:, None] * row_k

    # ---- public API -------------------------------------------------------
    def step(self, X: np.ndarray, active: Sequence[int] | None = None) -> np.ndarray:
        """Process one observed frame and return the separated frame.

        Parameters
        ----------
        X : ndarray, shape (F, K)
            Observed mixture x_ft for this frame, one row per bin.
        active : sequence of int, optional
            The index set I_k of sources to steer (Sec. IV-B). Default: all
            sources. Pass a subset once the estimate has converged and only
            those sources are moving.

        Returns
        -------
        Y : ndarray, shape (F, K)
            Separated frame y_ft = W_ft x_ft, computed after the update.
        """
        X = np.asarray(X, dtype=complex)
        if X.shape != (self.F, self.K):
            raise ValueError(f"X must have shape {(self.F, self.K)}, got {X.shape}")

        idx = range(self.K) if active is None else tuple(active)

        # x_ft x_ft^H depends on neither k nor the sweep index: form it once.
        outer = X[:, :, None] * np.conj(X[:, None, :])       # (F, K, K)

        # Anchor Eq. 9 to U_{t-1} (see module docstring). _update_covariances
        # rebinds self.U to a fresh array, so this snapshot stays intact.
        u_prev = self.U

        for _ in range(self.n_iter):
            self._update_covariances(u_prev, outer, X)
            self._update_demixing(idx)

        return np.einsum("fij,fj->fi", self.W, X)

    def demix(self, X: np.ndarray) -> np.ndarray:
        """Separate a frame with the current W, without updating anything."""
        return np.einsum("fij,fj->fi", self.W, np.asarray(X, dtype=complex))

    def project_back(self, Y: np.ndarray, ref_mic: int = 0) -> np.ndarray:
        """Minimal-distortion scaling (Murata et al., 2001).

        AuxIVA leaves each source's scale arbitrary; this rescales output k to
        how it would appear at microphone `ref_mic`. Needs one inverse of W
        per frame — it is an evaluation/output step, outside the update, and
        should not be counted against the inverse-free claim.
        """
        scale = np.linalg.inv(self.W)[:, ref_mic, :]         # (F, K)
        return Y * scale


# ---- WAV I/O (stdlib `wave`, no scipy/soundfile) ----------------------------
def _read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    """16-bit PCM mono WAV -> (sample_rate, float64 samples in [-1, 1])."""
    with wave.open(str(path), "rb") as f:
        fs = f.getframerate()
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        raw = f.readframes(f.getnframes())
    if sampwidth != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {8 * sampwidth}-bit")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)   # downmix to mono
    return fs, x


def _write_wav_mono(path: Path, fs: int, x: np.ndarray) -> None:
    """float samples (any range) -> 16-bit PCM mono WAV, peak-normalized."""
    peak = np.max(np.abs(x))
    x = x / peak * 0.98 if peak > 0 else x
    pcm = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes(pcm.tobytes())


# ---- STFT / ISTFT (NumPy only: Hann window, weighted overlap-add) ----------
def _stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Real signal -> (n_frames, n_freq) complex spectrogram."""
    window = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * window[None, :], axis=1)


def _istft(X: np.ndarray, n_fft: int, hop: int, length: int | None = None) -> np.ndarray:
    """(n_frames, n_freq) complex spectrogram -> real signal, WOLA reconstruction."""
    window = np.hanning(n_fft)
    n_frames = X.shape[0]
    out_len = n_fft + (n_frames - 1) * hop
    y = np.zeros(out_len)
    w_sum = np.zeros(out_len)
    frames_time = np.fft.irfft(X, n=n_fft, axis=1) * window[None, :]
    for i in range(n_frames):
        y[i * hop: i * hop + n_fft] += frames_time[i]
        w_sum[i * hop: i * hop + n_fft] += window ** 2
    # A Hann window tapers to exactly 0 at both edges, so only the first/last
    # few output samples (where a single frame's near-zero tail is the only
    # contributor) have a genuinely tiny w_sum — not zero, so the EPS guard
    # doesn't catch it, but small enough (~1e-9) that dividing by it turns
    # ordinary rounding noise in the numerator into a huge spurious spike.
    # That spike is a couple of samples out of tens of thousands, but it's
    # the loudest sample in the file: peak-normalizing against it silently
    # crushes every real sample by the same huge factor, and the file just
    # sounds dead. Clamp the floor relative to the steady-state (fully
    # overlapped) level rather than to an absolute epsilon — this costs a
    # bit of accuracy in the outermost ~1 hop of samples (~16 ms), same as
    # any windowed OLA scheme's edge behavior, not a new distortion.
    floor = max(EPS, 0.5 * w_sum.max())
    y = y / np.maximum(w_sum, floor)
    return y[:length] if length is not None else y


def _fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution via FFT (NumPy only). Length len(x) + len(h) - 1.

    Used to apply a room impulse response (thousands of taps) to a full
    source recording (hundreds of thousands of samples); direct convolution
    there is O(len(x) * len(h)) and much too slow.
    """
    n = len(x) + len(h) - 1
    n_fft = 1 << (n - 1).bit_length()   # next power of 2, for fast FFT
    return np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)[:n]


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR (dB), Le Roux et al. 2019 — self-contained metric
    (no mir_eval/scipy dependency) used only to sanity-check the demo output.
    """
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = (estimate @ reference) / (reference @ reference + EPS)
    projection = alpha * reference
    noise = estimate - projection
    return 10.0 * np.log10((projection @ projection + EPS) / (noise @ noise + EPS))


# ---- demo: separate two real speech recordings with 2 mics / 2 sources -----
def _demo():
    """Mix two real speech recordings through a simulated room (pyroomacoustics
    RIRs, RT60 = 0.2 s) plus additive noise (30 dB SNR), then separate them
    online with AuxIVA-ISS, in the determined K = 2 (sources = mics) case.

    The room, the two source/mic positions, and the RT60 are simulated —
    getting real reverberation would need an actual room and loudspeakers,
    not just a laptop — but pyroomacoustics generates genuine room impulse
    responses via the image-source method, so the mixing is a real
    convolutive (multipath) mixture, not the earlier version's anechoic
    pure-delay approximation. And the audio being mixed is real speech:
    audio/speaker1.wav and audio/speaker2.wav (two different macOS `say`
    voices reading different text, 16 kHz mono — see audio/README.txt), not
    synthetic complex-Gaussian spectra.

    Reverberation this long relative to the STFT window (RT60 = 0.2 s is
    3200 samples at 16 kHz; even a 1024-sample/64 ms window covers under a
    fifth of that) stresses AuxIVA-ISS's narrowband assumption — that
    convolutive mixing is approximately per-bin instantaneous multiplication
    in the STFT domain — more than the anechoic case did. Expect separation
    to be measurably harder here than in the pre-reverberation version of
    this demo; that is the reverberant per-bin approximation being imperfect,
    not a new bug, and matches the degree of imperfection AuxIVA-ISS's own
    paper reports at a similar RT60 (Sec. V).
    """
    audio_dir = Path(__file__).parent / "audio"
    out_dir = audio_dir / "output"
    out_dir.mkdir(exist_ok=True)
    src_paths = [audio_dir / "speaker1.wav", audio_dir / "speaker2.wav"]
    missing = [p for p in src_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing source audio: {missing}. On macOS you can regenerate them "
            f"with the `say` command, e.g.:\n"
            f'  say -v Daniel -o {src_paths[0]} --file-format=WAVE '
            f'--data-format=LEI16@16000 "some sentence"\n'
            f"or drop your own 16-bit PCM mono WAV files at those paths."
        )

    K = 2                      # 2 sources = 2 microphones (determined case)
    N_FFT, HOP = 1024, 512      # 64 ms / 50% overlap — the paper's own window
    # size, sized up from the anechoic version's 512/256 to give the
    # narrowband approximation the best reasonable chance against RT60=0.2s
    # reverberation (still well short of covering the full RIR — see above).
    ALPHA, N_ITER = 0.99, 2
    MIC_SPACING = 0.04          # m, a small 2-element array
    REF_MIC = 0
    RT60 = 0.4                  # s, target reverberation time
    SNR_DB = 30.0                # target SNR of the mic signal vs. added noise
    ROOM_DIM = [6.0, 5.0, 3.0]    # m
    MIC_CENTER = np.array([3.0, 2.0, 1.5])
    SRC_POS = [np.array([1.5, 3.8, 1.6]), np.array([4.8, 1.0, 1.4])]
    SEED = 0                    # for the additive noise only (everything
    # else here — the room, the RIRs, the audio — is deterministic)

    ACTIVE_MIN_HZ = 200.0
    ACTIVE_MAX_HZ = 0.9 * 343.0 / (2.0 * MIC_SPACING)   # ~3.86 kHz here.
    # Spatial-aliasing limit for this array spacing (c / 2d, with a 10%
    # margin): above it, different directions alias to the same inter-mic
    # phase, giving severely ill-conditioned per-bin mixing (cond numbers in
    # the hundreds to thousands were measured well above this cutoff on real
    # speech in the anechoic version of this demo) — a property of the array
    # geometry, present with or without reverberation. The low cutoff is a
    # gentler safety margin against the same narrowband-bin-conditioning
    # concern at the other end; with real (non-pure-delay) RIRs, per-source
    # DC gains generally differ, so DC is no longer exactly singular the way
    # it was in the anechoic model, but very low frequencies still carry
    # little directional information for a 4 cm array.

    print(f"loading {[p.name for p in src_paths]}...")
    fs_list, sources = zip(*(_read_wav_mono(p) for p in src_paths))
    if len(set(fs_list)) != 1:
        raise ValueError(f"sample rates differ: {fs_list}")
    fs = fs_list[0]
    n_samples = min(len(s) for s in sources)
    sources = [s[:n_samples] for s in sources]
    # Level-match the two recordings (equal RMS). Two independently recorded
    # sources are rarely at the same loudness, and AuxIVA's source model has
    # no reason to treat them symmetrically if they aren't — the louder one
    # tends to get separated cleanly while the quieter one lags, which is a
    # recording-level artifact rather than anything about the algorithm.
    rms = [np.sqrt(np.mean(s ** 2)) for s in sources]
    target_rms = float(np.mean(rms))
    sources = [s * (target_rms / r) for s, r in zip(sources, rms)]

    print(f"simulating room (RT60={RT60}s) and computing RIRs...")
    e_absorption, max_order = pra.inverse_sabine(RT60, ROOM_DIM)
    room = pra.ShoeBox(ROOM_DIM, fs=fs, materials=pra.Material(e_absorption),
                        max_order=max_order)
    mic_pos = MIC_CENTER[:, None] + MIC_SPACING * np.array([[-0.5, 0.5], [0, 0], [0, 0]])
    room.add_microphone_array(pra.MicrophoneArray(mic_pos, fs))
    for pos in SRC_POS:
        room.add_source(pos)
    room.compute_rir()   # room.rir[mic_idx][src_idx] -> 1D impulse response
    print(f"  absorption={e_absorption:.3f}  max_order={max_order}  "
          f"RIR length={len(room.rir[0][0]) / fs * 1000:.0f}ms")

    # Convolve each dry source with its own RIR at each mic, to get that
    # source's individual reverberant IMAGE at each mic — needed both to
    # build the mixture (their sum) and, at REF_MIC, as the evaluation
    # ground truth (comparing separated output to the dry, non-reverberant
    # source would penalize AuxIVA-ISS for reverberation it can't remove;
    # BSS evaluation always scores against the source's own room response).
    print("convolving sources with room impulse responses...")
    ref_img = [[_fft_convolve(sources[k], room.rir[i][k]) for i in range(K)]
               for k in range(K)]                       # ref_img[src][mic]
    n_conv = min(len(ref_img[k][i]) for k in range(K) for i in range(K))
    ref_img = [[r[:n_conv] for r in row] for row in ref_img]
    mix_clean = [sum(ref_img[k][i] for k in range(K)) for i in range(K)]  # per mic

    rng = np.random.default_rng(SEED)
    noisy_mix = []
    for clean in mix_clean:
        sig_power = np.mean(clean ** 2)
        noise_power = sig_power / (10.0 ** (SNR_DB / 10.0))
        noise = rng.standard_normal(len(clean)) * np.sqrt(noise_power)
        noisy_mix.append(clean + noise)
    print(f"  added noise for {SNR_DB} dB SNR "
          f"(measured: {10 * np.log10(np.mean(mix_clean[0]**2) / np.mean((noisy_mix[0]-mix_clean[0])**2)):.1f} dB)")

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / fs)
    active = (freqs >= ACTIVE_MIN_HZ) & (freqs <= ACTIVE_MAX_HZ)

    print("computing microphone-signal STFTs...")
    X_full = np.stack([_stft(x, N_FFT, HOP) for x in noisy_mix], axis=2)   # (T, F, K_mic)
    n_frames = X_full.shape[0]
    X_active = X_full[:, active, :]                                        # (T, F_active, K_mic)

    print(f"running online AuxIVA-ISS over {n_frames} frames "
          f"({n_frames * HOP / fs:.1f}s, {active.sum()} active bins)...")
    # Real speech has genuine silences (sentence pauses); synthetic bursty
    # envelopes never truly go quiet, so this doesn't show up in a purely
    # synthetic demo. In a near-silent frame r_kt -> 0, and the Laplace
    # weight phi(r) = 1/(2r) blows up, injecting a huge, meaningless update
    # into W that the online recursion doesn't recover from on its own — a
    # real robustness issue, not an artifact of this array/geometry. Gate
    # adaptation on frame energy (a minimal voice-activity proxy): during
    # silence, demix with the current W but don't update it.
    frame_energy = np.sum(np.abs(X_active) ** 2, axis=(1, 2))
    energy_gate = 0.01 * np.median(frame_energy)
    sep = OnlineAuxIVA_ISS(n_src=K, n_freq=int(active.sum()), alpha=ALPHA, n_iter=N_ITER)
    Y_full = np.zeros_like(X_full)                                    # back-projected output
    n_gated = 0
    for t in range(n_frames):
        if frame_energy[t] < energy_gate:
            y = sep.demix(X_active[t])
            n_gated += 1
        else:
            y = sep.step(X_active[t])
        Y_full[t, active, :] = sep.project_back(y, ref_mic=REF_MIC)
    print(f"  ({n_gated}/{n_frames} frames gated as near-silent, not adapted on)")
    # Bins outside [ACTIVE_MIN_HZ, ACTIVE_MAX_HZ] are never separated (see
    # above) and are left at zero rather than passed through unseparated.
    # Passing the raw mixture through looks fuller on casual playback, but
    # it's the *same* unseparated content in both output channels, and it
    # measurably inflates correlation/SDR against either source without
    # reflecting anything AuxIVA-ISS actually did — found by comparing the
    # two directly on this file's own output. Zeroing costs some bandwidth
    # (telephone-quality output, similar to the ~200 Hz - 3.4 kHz band POTS
    # used) but the result reflects the algorithm, not a metric artifact.

    print("computing ISTFTs and writing WAV files...")
    mix_time = _istft(X_full[:, :, REF_MIC], N_FFT, HOP, length=n_conv)
    sep_time = [_istft(Y_full[:, :, k], N_FFT, HOP, length=n_conv) for k in range(K)]
    # _stft drops a trailing partial frame, so the ISTFT round-trip can come
    # back a few samples shorter than n_conv (length= only truncates, it
    # can't pad past what the frames actually cover) — trim everything used
    # below, including the reverberant reference images, to whatever length
    # is actually common.
    n_eval = min(len(mix_time), *(len(y) for y in sep_time), n_conv)
    mix_time, sep_time = mix_time[:n_eval], [y[:n_eval] for y in sep_time]
    ref_eval = [ref_img[k][REF_MIC][:n_eval] for k in range(K)]
    _write_wav_mono(out_dir / "mix_mic0.wav", fs, mix_time)
    for k in range(K):
        _write_wav_mono(out_dir / f"separated_{k}.wav", fs, sep_time[k])
    print(f"wrote {out_dir}/mix_mic0.wav, separated_0.wav, separated_1.wav")

    # ---- evaluation: resolve the output permutation, then score it --------
    # Score each separated output against source k's own reverberant image
    # at REF_MIC (ref_eval), not the dry source: comparing to the dry source
    # would penalize AuxIVA-ISS for reverberation it was never going to
    # remove (it's a separator, not a dereverberation method), and that's
    # not the failure mode this demo is meant to show.
    #
    # Skip the first 20% of the recording: the online estimate starts at
    # W = I and needs a few seconds to converge, and scoring that warm-up
    # transient together with the converged steady state understates what
    # the algorithm settles into (confirmed on the anechoic version of this
    # demo by checking sliding 5s-window correlations directly — the early
    # windows were consistently much worse than everything after them).
    warmup = n_eval // 5
    ref_eval_w = [r[warmup:] for r in ref_eval]
    sep_eval = [y[warmup:] for y in sep_time]
    mix_eval = mix_time[warmup:]

    # AuxIVA has no constraint tying output channel k to source k; find
    # whichever assignment correlates best (K=2 -> only 2 permutations).
    corr = np.array([[np.corrcoef(ref_eval_w[j], sep_eval[k])[0, 1] for k in range(K)]
                      for j in range(K)])
    perm = (0, 1) if corr[0, 0] + corr[1, 1] >= corr[0, 1] + corr[1, 0] else (1, 0)
    print(f"\nresolved output permutation: source k -> output channel {list(perm)}")

    print(f"\nscored on the post-warm-up {(n_eval - warmup) / fs:.1f}s "
          f"(first {warmup / fs:.1f}s excluded as convergence transient), "
          f"against each source's own reverberant image at mic {REF_MIC}:")
    print(f"{'source':<8} {'baseline SI-SDR (dB)':>22} {'separated SI-SDR (dB)':>22}")
    for j in range(K):
        est = sep_eval[perm[j]]
        baseline = _si_sdr(ref_eval_w[j], mix_eval)
        separated = _si_sdr(ref_eval_w[j], est)
        print(f"{j:<8} {baseline:22.2f} {separated:22.2f}")


if __name__ == "__main__":
    _demo()
