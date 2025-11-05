import os, glob, pickle, argparse, numpy as np, mne, math
from loguru import logger
from config import ALL_CHANNELS, PATH_TO_RAW_DATA, PATH_TO_PROCESSED_DATA
from scipy.signal import resample as scipy_resample

# Memory-safe: no raw preload, per-epoch processing, optional decimation

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--chunk_duration", type=float, default=30.0)
    p.add_argument("--target_sampling_rate", type=int, default=256)
    p.add_argument("--pattern", type=str, default="RBD*.edf", help="Glob pattern (e.g., RBD12.edf or RBD*.edf)")
    p.add_argument("--max_files", type=int, default=-1, help="Limit number of EDFs (for quick tests)")
    return p.parse_args()

def _safe_pick(raw):
    # ensure all requested channels exist
    missing = [ch for ch in ALL_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise RuntimeError(f"Missing required channels: {missing}")

    # pick by integer indices (compatible with older MNE)
    picks = [raw.ch_names.index(ch) for ch in ALL_CHANNELS]
    raw.pick(picks)

    # reorder explicitly to match ALL_CHANNELS order
    raw.rename_channels({name: name for name in raw.ch_names})  # no-op, just safe
    raw.reorder_channels(ALL_CHANNELS)


def _epoch_iterator(raw, chunk_duration, target_fs):
    """Yield (idx, epoch_data [n_chan x n_samples float32], label_str) without loading all epochs."""
    orig_fs = raw.info["sfreq"]
    events = mne.make_fixed_length_events(raw, duration=chunk_duration)

    # If target_fs divides orig_fs nicely, use decimation (cheap and memory-light)
    decim = None
    if abs(orig_fs / target_fs - round(orig_fs / target_fs)) < 1e-6:
        decim = int(round(orig_fs / target_fs))

    epochs = mne.Epochs(
        raw, events,
        tmin=0.0, tmax=chunk_duration - 1.0 / orig_fs,
        baseline=None,
        preload=False,            # keep memory low
        verbose=False,
        decim=decim,
        reject_by_annotation='omit'
    )

    n_events = epochs.events.shape[0]  # don't use len(epochs) with preload=False

    for i in range(n_events):
        # epochs[i] is a 1-epoch Epochs object; get_data() loads just that epoch
        x = epochs[i].get_data()[0]  # shape (n_chan, n_samples), no 'verbose' kwarg on older MNE
        if decim is None and target_fs != orig_fs:
            new_len = int(round(x.shape[1] * (target_fs / orig_fs)))
            from scipy.signal import resample as scipy_resample
            x = scipy_resample(x, num=new_len, axis=1)
        yield i, x.astype(np.float32), "W"  # dummy label



def main():
    args = get_args()
    raw_dir = args.data_path or PATH_TO_RAW_DATA
    out_dir = args.save_path or PATH_TO_PROCESSED_DATA
    os.makedirs(out_dir, exist_ok=True)
    x_dir = os.path.join(out_dir, "X")
    y_dir = os.path.join(out_dir, "Y")
    os.makedirs(x_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, args.pattern)))
    if args.max_files != -1:
        files = files[:args.max_files]
    logger.info(f"Found {len(files)} EDF files matching '{args.pattern}'")

    for edf in files:
        base = os.path.splitext(os.path.basename(edf))[0]
        logger.info(f"Processing {base}")
        try:
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")  # <-- preload=False
            _safe_pick(raw)
        except Exception as e:
            logger.error(f"Skipping {base}. Reason: {e}")
            continue

        X_patient_dir = os.path.join(x_dir, base)
        os.makedirs(X_patient_dir, exist_ok=True)
        labels = {}

        try:
            for i, ep, lab in _epoch_iterator(raw, args.chunk_duration, args.target_sampling_rate):
                fname = f"{base}_{i}.npy"
                np.save(os.path.join(X_patient_dir, fname), ep)
                labels[fname] = lab
        except Exception as e:
            logger.error(f"Failed while epoching {base}. Reason: {e}")
            if len(os.listdir(X_patient_dir)) == 0:
                try: os.rmdir(X_patient_dir)
                except: pass
            continue

        if len(labels) > 0:
            with open(os.path.join(y_dir, f"{base}.pickle"), "wb") as f:
                pickle.dump(labels, f)
            logger.info(f"Saved {len(labels)} epochs for {base}")
        else:
            logger.warning(f"No epochs saved for {base}; removing empty folder.")
            try: os.rmdir(X_patient_dir)
            except: pass

    logger.info("✅ Extraction complete (memory-safe).")

if __name__ == "__main__":
    main()
