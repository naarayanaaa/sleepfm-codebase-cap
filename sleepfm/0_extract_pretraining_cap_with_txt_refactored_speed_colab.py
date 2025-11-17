import os
import glob
import pickle
import argparse
import re
from typing import List, Optional, Tuple

import numpy as np
import mne
from loguru import logger
from scipy.signal import resample as scipy_resample

from config import ALL_CHANNELS, PATH_TO_RAW_DATA, PATH_TO_PROCESSED_DATA

# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--chunk_duration", type=float, default=30.0)
    p.add_argument("--target_sampling_rate", type=int, default=256)
    p.add_argument(
        "--pattern",
        type=str,
        default="*.edf",
        help="Glob pattern for EDFs (e.g., RBD12.edf or RBD*.edf)",
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=-1,
        help="Limit number of EDFs (for quick tests)",
    )
    return p.parse_args()


# -------------------------------------------------------------------------
# Channel picking helper
# -------------------------------------------------------------------------


def _safe_pick(raw: mne.io.BaseRaw):
    """Ensure all requested channels exist and are ordered as ALL_CHANNELS."""
    missing = [ch for ch in ALL_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise RuntimeError(f"Missing required channels: {missing}")

    picks = [raw.ch_names.index(ch) for ch in ALL_CHANNELS]
    raw.pick(picks)
    raw.rename_channels({name: name for name in raw.ch_names})
    raw.reorder_channels(ALL_CHANNELS)


# -------------------------------------------------------------------------
# CAP TXT parsing helpers
# -------------------------------------------------------------------------

_TIME_RE = re.compile(r"^\d{1,2}[:.]\d{2}[:.]\d{2}$")


def _parse_time_to_seconds(t: str) -> Optional[int]:
    """
    Parse time like '22:18:17' or '22.18.17' into seconds since midnight.
    Returns None if parsing fails.
    """
    t = t.strip()
    sep = ":" if ":" in t else "."
    parts = t.split(sep)
    if len(parts) != 3:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def _cap_stage_to_label(stage_code: str) -> Optional[str]:
    """
    Map CAP sleep stage codes to SleepFM label strings that work with LABEL_MAP.

    Returns one of: "Wake", "Stage 1", "Stage 2", "Stage 3", "REM"
    or None if we want to drop it.
    """
    if not stage_code:
        return None
    c = stage_code.strip().upper()

    if c in {"W", "WAKE"}:
        return "Wake"
    if c in {"R", "REM"}:
        return "REM"
    if c in {"S1", "N1"}:
        return "Stage 1"
    if c in {"S2", "N2"}:
        return "Stage 2"
    if c in {"S3", "S4", "N3"}:
        return "Stage 3"
    if c in {"MT", "MOVEMENT"}:
        # Movement time is often treated as wake in many analyses
        return "Wake"

    # Anything unknown we drop rather than crash later
    return None


def _find_header_line_index(lines: List[str]) -> Optional[int]:
    """
    Find the line index where the header starts (line beginning with 'Sleep Stage').
    """
    for i, line in enumerate(lines):
        if line.strip().startswith("Sleep Stage"):
            return i
    return None


def _parse_cap_txt_file(txt_path: str, chunk_duration: float) -> Optional[List[str]]:
    """
    Parse a CAP .txt scoring file into a list of per-epoch labels.

    Strategy (robust but simple, no pandas dependency here):
    - Find header line ('Sleep Stage ...').
    - For each subsequent non-empty line:
        * First token: stage code (W, S1, S2, S3, R, etc.).
        * Find token that looks like a time (hh:mm:ss or hh.mm.ss).
        * Token after time: event label; we keep only rows whose event looks like sleep staging
          (contains 'SLEEP-S' or looks like standard sleep stages).
        * Next token: duration in seconds (usually 30).
    - Build a sequential list of labels, replicating each label for duration/chunk_duration epochs.

    Returns:
        labels: List[str] of epoch labels ("Wake", "Stage 1", ...),
        or None if parsing completely failed.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read TXT file {txt_path}: {e}")
        return None

    header_idx = _find_header_line_index(lines)
    if header_idx is None:
        logger.error(f"Could not find 'Sleep Stage' header in {txt_path}")
        return None

    stage_rows: List[Tuple[str, Optional[int], float]] = []  # (label, time_sec, duration)

    prev_time_sec: Optional[int] = None
    sanity_mismatch_count = 0

    for line in lines[header_idx + 1 :]:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 3:
            continue

        stage_code = parts[0]
        label_str = _cap_stage_to_label(stage_code)
        if label_str is None:
            # Skip things we don't know how to handle
            continue

        # Find time token
        time_idx = None
        time_sec = None
        for j, tok in enumerate(parts):
            if _TIME_RE.match(tok.strip()):
                time_idx = j
                time_sec = _parse_time_to_seconds(tok)
                break

        if time_idx is None:
            # Cannot locate time, but we can still use duration as 30s and keep order
            time_sec = None

        # Event and duration are typically after the time token
        event_tok = None
        dur_tok = None
        if time_idx is not None:
            if time_idx + 1 < len(parts):
                event_tok = parts[time_idx + 1]
            if time_idx + 2 < len(parts):
                dur_tok = parts[time_idx + 2]
        else:
            # Fallback: assume last-but-two and last-but-one positions
            if len(parts) >= 3:
                event_tok = parts[-3]
                dur_tok = parts[-2]

        # Filter to keep only sleep-stage rows. CAP A-phase rows have MCAP-A1/A2/A3 etc.
        if event_tok is not None and "MCAP-A" in event_tok.upper():
            # This is an A-phase CAP event, not baseline sleep stage; skip
            continue

        # Duration
        dur_sec = 30.0
        if dur_tok is not None:
            try:
                dur_sec = float(dur_tok.replace(",", "."))
            except Exception:
                dur_sec = 30.0

        # Sanity check: does time difference roughly match duration?
        if prev_time_sec is not None and time_sec is not None:
            dt = time_sec - prev_time_sec
            # handle wrap across midnight (rare but possible)
            if dt < 0:
                dt += 24 * 3600
            if abs(dt - prev_time_sec_duration) > 5.0:  # 5 seconds tolerance
                sanity_mismatch_count += 1

        stage_rows.append((label_str, time_sec, dur_sec))
        prev_time_sec = time_sec
        prev_time_sec_duration = dur_sec

    if not stage_rows:
        logger.error(f"No usable sleep-stage rows parsed from {txt_path}")
        return None

    if sanity_mismatch_count > 0:
        logger.warning(
            f"[{os.path.basename(txt_path)}] "
            f"{sanity_mismatch_count} time/duration mismatches encountered while parsing."
        )

    # Convert event rows into **epoch-wise labels**:
    labels: List[str] = []
    for label_str, _time_sec, dur_sec in stage_rows:
        n_epochs = max(1, int(round(dur_sec / float(chunk_duration))))
        labels.extend([label_str] * n_epochs)

    # Keep this info-level log relatively light
    logger.info(
        f"[{os.path.basename(txt_path)}] Parsed {len(stage_rows)} stage rows "
        f"into {len(labels)} {chunk_duration:.0f}s epoch labels."
    )
    return labels


# -------------------------------------------------------------------------
# Epoch iterator (optimized, but same external behaviour)
# -------------------------------------------------------------------------


def _epoch_iterator(
    raw: mne.io.BaseRaw,
    chunk_duration: float,
    target_fs: int,
    epoch_labels: Optional[List[str]] = None,  # kept for signature compatibility, not used here
):
    """
    Yield (idx, epoch_data [n_chan x n_samples float32], label_str_placeholder).

    - Uses fixed-length epochs over the entire recording.
    - Resamples/decimates to target_fs.
    - Label alignment is handled by the caller (we always yield None for label_str).
    """
    orig_fs = float(raw.info["sfreq"])

    # Fixed-length events over the whole night
    events = mne.make_fixed_length_events(raw, duration=chunk_duration)

    # If target_fs divides orig_fs nicely, use decimation inside MNE
    decim = None
    if abs(orig_fs / target_fs - round(orig_fs / target_fs)) < 1e-6:
        decim = int(round(orig_fs / target_fs))

    # Preload all epochs into memory once
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=chunk_duration - 1.0 / orig_fs,
        baseline=None,
        preload=True,
        verbose=False,
        decim=decim,
        reject_by_annotation="omit",
    )

    data = epochs.get_data()  # shape: (n_events, n_chan, n_samples_at_current_fs)
    n_events, n_chan, n_samp = data.shape

    # If we did not decimate and fs doesn't match, resample ONCE over all epochs
    if decim is None and target_fs != orig_fs:
        new_len = int(round(n_samp * (target_fs / orig_fs)))
        # Resample along the time axis (axis=2) for all epochs at once
        data = scipy_resample(data, num=new_len, axis=2)

    # Yield per-epoch arrays as before
    for i in range(data.shape[0]):
        x = data[i]  # (n_chan, n_samples)
        yield i, x.astype(np.float32), None  # labels handled outside


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main():
    args = get_args()
    raw_dir = args.data_path or PATH_TO_RAW_DATA
    out_dir = args.save_path or PATH_TO_PROCESSED_DATA

    # We still allow changing this via CLI, but you keep it at 30s
    chunk_duration = float(args.chunk_duration)
    target_fs = int(args.target_sampling_rate)

    os.makedirs(out_dir, exist_ok=True)
    x_dir = os.path.join(out_dir, "X")
    y_dir = os.path.join(out_dir, "Y")
    os.makedirs(x_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, args.pattern)))
    if args.max_files != -1:
        files = files[: args.max_files]

    logger.info(f"Found {len(files)} EDF files matching '{args.pattern}' in {raw_dir}")

    for edf in files:
        base = os.path.splitext(os.path.basename(edf))[0]
        X_patient_dir = os.path.join(x_dir, base)
        y_pickle_path = os.path.join(y_dir, f"{base}.pickle")

        # -----------------------------------------------------------------
        # Simple optimization: skip if this patient is already processed
        # (same outputs, just avoids recomputing work).
        # -----------------------------------------------------------------
        if os.path.exists(y_pickle_path) and os.path.isdir(X_patient_dir) and len(os.listdir(X_patient_dir)) > 0:
            logger.info(f"{base}: already processed, skipping.")
            continue

        logger.info(f"Processing {base}")

        # -----------------------------------------------------------------
        # Read EDF (preload into RAM once)
        # -----------------------------------------------------------------
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
            _safe_pick(raw)
        except Exception as e:
            logger.error(f"Skipping {base}. EDF read/pick failed: {e}")
            continue

        # -----------------------------------------------------------------
        # Locate matching TXT file and parse labels
        # -----------------------------------------------------------------
        txt_path = os.path.join(os.path.dirname(edf), f"{base}.txt")
        if not os.path.exists(txt_path):
            logger.warning(f"{base}: no TXT scoring file found; skipping because labels are missing.")
            continue

        epoch_labels = _parse_cap_txt_file(txt_path, chunk_duration=chunk_duration)
        if epoch_labels is None or len(epoch_labels) == 0:
            logger.warning(f"{base}: TXT parsing yielded no labels; skipping.")
            continue

        # -----------------------------------------------------------------
        # Epoching and saving
        # -----------------------------------------------------------------
        os.makedirs(X_patient_dir, exist_ok=True)
        labels_dict = {}

        try:
            # Generate all epochs (fast, single preload + resample)
            epochs_iter = list(
                _epoch_iterator(
                    raw,
                    chunk_duration,
                    target_fs,
                    epoch_labels=None,  # labels handled manually
                )
            )

            n_epochs_edf = len(epochs_iter)
            n_labels = len(epoch_labels)
            n_use = min(n_epochs_edf, n_labels)

            # Compute how many we drop on each side (for logging)
            dropped_edf = max(0, n_epochs_edf - n_use)
            dropped_txt = max(0, n_labels - n_use)

            if n_use == 0:
                logger.warning(
                    f"{base}: EDF epochs={n_epochs_edf}, TXT labels={n_labels}. "
                    f"No overlapping epochs -> nothing saved."
                )
            else:
                # Save only the matching pairs
                for i in range(n_use):
                    _, ep, _ = epochs_iter[i]
                    lab = epoch_labels[i]
                    fname = f"{base}_{i}.npy"
                    np.save(os.path.join(X_patient_dir, fname), ep)
                    labels_dict[fname] = lab

        except Exception as e:
            logger.error(f"Failed while epoching {base}. Reason: {e}")
            # Clean up empty folder if nothing was saved
            if os.path.isdir(X_patient_dir) and len(os.listdir(X_patient_dir)) == 0:
                try:
                    os.rmdir(X_patient_dir)
                except Exception:
                    pass
            continue

        # -----------------------------------------------------------------
        # Save labels for this patient
        # -----------------------------------------------------------------
        if len(labels_dict) > 0:
            with open(y_pickle_path, "wb") as f:
                pickle.dump(labels_dict, f)

            logger.info(
                f"{base}: saved {len(labels_dict)} epochs "
                f"(EDF epochs: {n_epochs_edf}, TXT labels: {n_labels}, "
                f"used: {n_use}, dropped EDF: {dropped_edf}, dropped TXT: {dropped_txt})"
            )
        else:
            logger.warning(f"{base}: no epochs saved; removing empty folder.")
            try:
                os.rmdir(X_patient_dir)
            except Exception:
                pass

    logger.info("✅ Extraction with CAP TXT labels complete.")


if __name__ == "__main__":
    main()
