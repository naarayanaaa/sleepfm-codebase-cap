import os, glob, pickle, argparse, numpy as np, mne
from loguru import logger
from config import ALL_CHANNELS, PATH_TO_RAW_DATA, PATH_TO_PROCESSED_DATA

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--chunk_duration", type=float, default=30.0)
    p.add_argument("--target_sampling_rate", type=int, default=256)
    return p.parse_args()

def main():
    args = get_args()
    raw_dir = args.data_path or PATH_TO_RAW_DATA
    out_dir = args.save_path or PATH_TO_PROCESSED_DATA
    os.makedirs(out_dir, exist_ok=True)
    x_dir = os.path.join(out_dir, "X")
    y_dir = os.path.join(out_dir, "Y")
    os.makedirs(x_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)

    edfs = sorted(glob.glob(os.path.join(raw_dir, "RBD*.edf")))
    logger.info(f"Found {len(edfs)} EDF files")

    for edf in edfs:
        base = os.path.splitext(os.path.basename(edf))[0]
        logger.info(f"Processing {base}")
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")

        # Ensure consistent channel order
        missing = [ch for ch in ALL_CHANNELS if ch not in raw.ch_names]
        if missing:
            logger.warning(f"Skipping {base}: missing channels {missing}")
            continue
        raw.pick_channels(ALL_CHANNELS)

        # Resample to target frequency
        if raw.info["sfreq"] != args.target_sampling_rate:
            raw.resample(args.target_sampling_rate)

        # Make 30s fixed-length epochs
        epochs = mne.make_fixed_length_epochs(raw, duration=args.chunk_duration, preload=True, verbose="ERROR")

        # Save each epoch as .npy
        X_patient_dir = os.path.join(x_dir, base)
        os.makedirs(X_patient_dir, exist_ok=True)

        labels = {}
        for i, ep in enumerate(epochs.get_data()):
            fname = f"{base}_{i}.npy"
            np.save(os.path.join(X_patient_dir, fname), ep.astype(np.float32))
            labels[fname] = "W"  # dummy label

        with open(os.path.join(y_dir, f"{base}.pickle"), "wb") as f:
            pickle.dump(labels, f)

    logger.info("✅ Extraction complete! Check your processed folder.")

if __name__ == "__main__":
    main()
