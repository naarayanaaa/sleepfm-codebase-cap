import glob
import os
from collections import Counter
import warnings

import mne

# Silence MNE runtime warnings about filters, etc.
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# Root of the CAP sleep database
raw_dir = r"C:\Users\JD\Downloads\cap-sleep-database-1.0.0\cap-sleep-database-1.0.0"

files = sorted(glob.glob(os.path.join(raw_dir, "*.edf")))
if not files:
    print("No EDF files found. Check folder path.")
    raise SystemExit

channel_counts = Counter()
file_channels = {}  # basename -> set of channels (uppercased)
n_files = 0

for f in files:
    basename = os.path.basename(f)
    try:
        raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
    except Exception as e:
        print(f"[!] Skipping {basename}: {e}")
        continue

    n_files += 1

    # Normalize labels: strip spaces and uppercase
    chs = set(c.strip().upper() for c in raw.ch_names)
    file_channels[basename] = chs
    channel_counts.update(chs)

print(f"\nScanned {n_files} EDF files from: {raw_dir}\n")

# Show channels by how often they occur
print("Top 40 channels by frequency:\n")
for ch, cnt in channel_counts.most_common(40):
    print(f"{ch:20s} {cnt:4d} files ({cnt / n_files:5.1%})")

# Channels that appear in at least X% of files
min_coverage = 0.80  # 80%; change this if you like
print(f"\nChannels present in at least {min_coverage*100:.0f}% of files:\n")
for ch, cnt in sorted(channel_counts.items(), key=lambda x: -x[1]):
    if cnt / n_files >= min_coverage:
        print(f"{ch:20s} {cnt:4d} files ({cnt / n_files:5.1%})")

# ----------------------------------------------------------------------
# Groups of channels and which samples share them
# ----------------------------------------------------------------------

groups = {
    "Group 1 (4 chans)": {
        "F4-C4", "P4-O2", "C4-P4", "C4-A1",
    },
    "Group 2 (8 chans)": {
        "F4-C4", "P4-O2", "C4-P4", "C4-A1",
        "FP2-F4", "ROC-LOC", "EMG1-EMG2", "ECG1-ECG2",
    },
    "Group 3 (11 chans)": {
        "F4-C4", "P4-O2", "C4-P4", "C4-A1",
        "FP2-F4", "ROC-LOC", "EMG1-EMG2", "ECG1-ECG2",
        "SX1-SX2", "DX1-DX2", "HR",
    },
}

print("\n====================================================================")
print("Samples containing each requested channel group")
print("====================================================================\n")

# Keep track of all files that match at least one group
files_in_any_group = set()

for group_name, ch_set in groups.items():
    matching_files = sorted(
        fname
        for fname, chs in file_channels.items()
        if ch_set.issubset(chs)
    )

    # Accumulate files that belong to at least one group
    files_in_any_group.update(matching_files)

    print(group_name)
    print("Required channels:", ", ".join(sorted(ch_set)))
    print(f"Number of matching EDFs: {len(matching_files)}")
    if matching_files:
        print("Matching EDF files:")
        for fname in matching_files:
            print("  ", fname)
    else:
        print("  [No EDF files contain all channels in this group]")
    print()

# ----------------------------------------------------------------------
# Files that are NOT in any of the groups
# ----------------------------------------------------------------------

remaining_files = sorted(set(file_channels.keys()) - files_in_any_group)

print("====================================================================")
print("Samples NOT included in any group (and their channels)")
print("====================================================================\n")

if not remaining_files:
    print("All EDF files belong to at least one of the defined groups.\n")
else:
    print(f"Number of EDFs not in any group: {len(remaining_files)}\n")
    for fname in remaining_files:
        chs = sorted(file_channels[fname])
        print(fname)
        print("  Channels:", ", ".join(chs))
        print()
