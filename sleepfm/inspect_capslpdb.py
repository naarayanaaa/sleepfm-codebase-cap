#!/usr/bin/env python
"""
inspect_capslpdb.py

Explore structure of PhysioNet CAP Sleep Database files:
  - EDF: channel names, sampling freq, duration (using mne.io.read_raw_edf)
  - TXT: REMlogic export with sleep macro/micro structure (manual parsing)
  - EDF.ST: PhysioBank WFDB annotations (if wfdb is installed)

Usage:
  python inspect_capslpdb.py --root C:\\path\\to\\cap-sleep-database-1.0.0
  python inspect_capslpdb.py --root ... --record brux1
"""

import argparse
import textwrap
from pathlib import Path

import mne

try:
    import wfdb
except ImportError:
    wfdb = None


# ---------------------------------------------------------------------
# EDF inspection using MNE
# ---------------------------------------------------------------------
def inspect_edf(edf_path: Path):
    print(f"\n=== EDF: {edf_path.name} ===")
    if not edf_path.exists():
        print("  [!] EDF file not found.")
        return

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    except Exception as e:
        print(f"  [!] mne.io.read_raw_edf failed: {e}")
        return

    ch_names = raw.ch_names
    sfreq = float(raw.info["sfreq"])
    duration_sec = raw.times[-1] if len(raw.times) > 0 else raw.n_times / sfreq

    print(f"  Number of channels: {len(ch_names)}")
    print(f"  Sampling frequency (global): {sfreq:.2f} Hz")
    print(f"  Recording duration: {duration_sec:.1f} s (~{duration_sec/3600:.2f} h)")
    print("  First 20 channel names:")
    for i, name in enumerate(ch_names[:20]):
        print(f"    [{i:02d}] {name}")
    if len(ch_names) > 20:
        print("    ...")


# ---------------------------------------------------------------------
# TXT inspection using logic from ScoringReader.m (no pandas)
# ---------------------------------------------------------------------
def inspect_txt(txt_path: Path):
    print(f"\n=== TXT scoring (REMlogic export): {txt_path.name} ===")
    if not txt_path.exists():
        print("  [!] TXT file not found.")
        return

    with txt_path.open("r", encoding="latin-1") as f:
        lines = [ln.rstrip("\n\r") for ln in f]

    # Find the header line: starts with "Sleep Stage"
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Sleep Stage"):
            header_idx = i
            break

    if header_idx is None:
        print("  [!] Could not find 'Sleep Stage' header in TXT file.")
        # Show first few lines for manual inspection
        print("  First 10 lines:")
        for ln in lines[:10]:
            print("   ", ln)
        return

    print(f"  Header line index: {header_idx}")
    print("  Header content:", lines[header_idx])

    # Now emulate the core of ScoringReader.m:
    # - hyp: list of (stage_code, absolute_time_sec)
    # - cap: list of CAP A phases (type, duration, absolute_time_sec, raw_line)
    hyp_entries = []
    cap_entries = []
    start_parsing = False
    k = 0  # hyp counter
    j = 0  # CAP counter

    for ln in lines[header_idx + 1:]:
        if not ln or len(ln) <= 10:
            continue

        # ScoringReader uses colon positions to parse time & type:
        # p = strfind(tline,':');
        # if numel(p)==0, p=strfind(tline,'.'); end
        colon_idx = [idx for idx, ch in enumerate(ln) if ch == ":"]
        if not colon_idx:
            # Some lines may use '.'; mimic the MATLAB fallback:
            colon_idx = [idx for idx, ch in enumerate(ln) if ch == "."]

        if len(colon_idx) < 2:
            # Not in the expected format, skip
            continue

        p1, p2 = colon_idx[0], colon_idx[1]

        # Guard for short lines
        if p2 + 6 > len(ln):
            continue

        token = ln[p2 + 4 : p2 + 6]  # 'SL' (Sleep Stage) or 'MC' (CAP microstructure)

        # -----------------------------------------------------------------
        # Sleep stage line (hypnogram)
        # -----------------------------------------------------------------
        if token == "SL":
            # Stage code: char at p2+11 (ScoringReader.m)
            if p2 + 12 > len(ln):
                continue
            stage_char = ln[p2 + 11]

            if stage_char == "E":       # REM
                stage_code = 5
            elif stage_char == "T":     # MT (movement time) = 7 in ScoringReader
                stage_code = 7
            else:
                try:
                    stage_code = int(stage_char)
                except ValueError:
                    # Non-standard stage char, store as None
                    stage_code = None

            # Time fields around p1,p2
            try:
                # hour: tline(p(1)-2:p(1)-1) in MATLAB (inclusive)
                hh = int(ln[p1 - 2 : p1])
                # minute: tline(p(1)+1:p(1)+2)
                mm = int(ln[p1 + 1 : p1 + 3])
                # second: tline(p(2)+1:p(2)+2)
                ss = int(ln[p2 + 1 : p2 + 3])
            except Exception:
                continue

            # Absolute time in seconds as in ScoringReader:
            if hh < 10:
                abs_t = (hh + 24) * 3600 + mm * 60 + ss
            else:
                abs_t = hh * 3600 + mm * 60 + ss

            k += 1
            hyp_entries.append(
                {
                    "stage_code": stage_code,  # 0–4, 5=REM, 7=MT
                    "hh": hh,
                    "mm": mm,
                    "ss": ss,
                    "abs_time_sec": abs_t,
                    "raw": ln,
                }
            )

        # -----------------------------------------------------------------
        # CAP A phase line (microstructure)
        # -----------------------------------------------------------------
        elif token == "MC":
            # MC lines encode CAP A-phase type and duration
            # ScoringReader.m:
            #   t = strfind(tline,'-');
            #   type_ar(j) = str2num(tline(t(1)+2));
            #   duration(j) = str2num(tline(t(1)+4:t(1)+5));
            d_idx = ln.find("-")
            if d_idx == -1 or d_idx + 6 > len(ln):
                continue

            type_char = ln[d_idx + 2]
            try:
                cap_type = int(type_char)  # 1,2,3
            except ValueError:
                cap_type = None

            try:
                dur_str = ln[d_idx + 4 : d_idx + 6]
                duration_sec = int(dur_str)
            except Exception:
                duration_sec = None

            # Time fields same as for hyp
            try:
                hh = int(ln[p1 - 2 : p1])
                mm = int(ln[p1 + 1 : p1 + 3])
                ss = int(ln[p2 + 1 : p2 + 3])
            except Exception:
                continue

            if hh < 10:
                abs_t = (hh + 24) * 3600 + mm * 60 + ss
            else:
                abs_t = hh * 3600 + mm * 60 + ss

            j += 1
            cap_entries.append(
                {
                    "cap_type": cap_type,       # 1,2,3 (A1,A2,A3)
                    "duration_sec": duration_sec,
                    "hh": hh,
                    "mm": mm,
                    "ss": ss,
                    "abs_time_sec": abs_t,
                    "raw": ln,
                }
            )
        else:
            # Some other kind of event, ignore for now
            continue

    # --------------------- summary output ---------------------
    print(f"  Parsed sleep stage entries: {len(hyp_entries)}")
    if hyp_entries:
        unique_stages = sorted({e['stage_code'] for e in hyp_entries})
        print(f"  Unique stage codes (0–4 NREM, 5 REM, 7 MT, None=unknown): {unique_stages}")
        print("  Example 3 sleep stage lines:")
        for e in hyp_entries[:3]:
            print("   ", e["raw"])

    print(f"  Parsed CAP A-phase entries: {len(cap_entries)}")
    if cap_entries:
        unique_cap_types = sorted({e['cap_type'] for e in cap_entries})
        print(f"  Unique CAP types (1=A1,2=A2,3=A3, None=unknown): {unique_cap_types}")
        print("  Example 3 CAP A-phase lines:")
        for e in cap_entries[:3]:
            print("   ", e["raw"])

    # If needed later, you can reuse hyp_entries and cap_entries
    # to build a full hypnogram and CAP time series in Python.


# ---------------------------------------------------------------------
# EDF.ST inspection (optional, wfdb-based)
# ---------------------------------------------------------------------
def inspect_st(st_path: Path) -> None:
    """
    Inspect <record>.edf.st annotations using wfdb.

    Example: st_path = Path(.../"brux1.edf.st")
    We use wfdb.rdann with record name '.../brux1.edf' and annotator 'st'.
    """
    print(f"\n=== EDF.ST annotations: {st_path.name} ===")

    try:
        import wfdb
    except ImportError:
        print("  [!] wfdb not installed; skipping EDF.ST inspection.")
        return

    try:
        # 'brux1.edf.st' -> 'brux1.edf'
        rec_name = str(st_path.with_suffix(""))   # remove only ".st"
        ann = wfdb.rdann(rec_name, "st")
    except Exception as e:
        print(f"  [!] wfdb.rdann failed: {e}")
        return

    print(f"  Number of annotation samples: {len(ann.sample)}")
    print(f"  Unique annotation symbols: {sorted(set(ann.symbol))}")
    print(f"  First 5 annotations (sample, symbol, aux_note):")
    for i in range(min(5, len(ann.sample))):
        print(f"    {ann.sample[i]}  {ann.symbol[i]!r}  {ann.aux_note[i]!r}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inspect CAP Sleep Database EDF/annotation file structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Notes:
              - RECORDS file is expected under --root and should list .edf files (one per line).
              - For each base name X in RECORDS, we look for X.edf, X.edf.st, X.txt.
              - TXT = REMlogic export (macro + micro structure) parsed using ScoringReader.m logic.
              - EDF.ST = PhysioBank WFDB annotations of same events (time in samples).
            """
        ),
    )
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of CAP Sleep Database (where RECORDS lives).")
    parser.add_argument("--record", type=str, default=None,
                        help="Optional: inspect only this record (e.g. 'brux1' or 'brux1.edf').")

    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    records_file = root / "RECORDS"
    if not records_file.exists():
        print(f"[!] RECORDS file not found at {records_file}")
        return

    with records_file.open("r") as f:
        record_lines = [ln.strip() for ln in f if ln.strip()]

    edf_names = record_lines

    if args.record is not None:
        target = args.record
        if not target.endswith(".edf"):
            target = target + ".edf"
        edf_names = [r for r in edf_names if r == target]
        if not edf_names:
            print(f"[!] Requested record {args.record} not found in RECORDS.")
            return

    for edf_name in edf_names:
        base = Path(edf_name).stem
        edf_path = root / f"{base}.edf"
        st_path = root / f"{base}.edf.st"
        txt_path = root / f"{base}.txt"

        print("\n" + "=" * 80)
        print(f"RECORD: {base}")
        print("=" * 80)
        inspect_edf(edf_path)
        inspect_txt(txt_path)
        inspect_st(st_path)


if __name__ == "__main__":
    main()
