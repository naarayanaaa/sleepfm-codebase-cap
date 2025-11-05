import glob, mne, os
mne.set_log_level("ERROR")

# EDIT THIS PATH IF YOUR RAW EDFs LIVE SOMEWHERE ELSE:
raw_dir = r"C:\Users\JD\sleepfm_cap_rbd\data\raw"

files = sorted(glob.glob(os.path.join(raw_dir, "RBD*.edf")))
if not files:
    print("No files found. Check the folder path and file names (e.g., RBD1.edf).")
    raise SystemExit

common = None
for f in files:
    raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
    ch = set([c.strip() for c in raw.ch_names])
    common = ch if common is None else common.intersection(ch)

print("Scanned", len(files), "EDF files from:", raw_dir)
print("Common channel labels across all files:\n", sorted(common))
