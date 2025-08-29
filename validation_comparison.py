import os, re
import pandas as pd
from collections import defaultdict
from typing import Set  # <-- NEW for Py ≤3.8 compatibility

# -------------------------------------------------------------------
# 1.  paths
# -------------------------------------------------------------------
BASE_DIR    = r"D:\final_chicago_results\therapists"
MAIN_CSV    = os.path.join(BASE_DIR, "therapist_1_updated.csv")
DETAIL_DIR  = os.path.join(BASE_DIR, "therapist_1")   # 100 per-trial CSVs
from collections import defaultdict, Counter       # add Counter

# ---------------- counters ----------------
diff_task = Counter()                              ### NEW
diff_seg  = defaultdict(Counter)                  ### NEW
# -------------------------------------------------------------------
# 2.  helpers
# -------------------------------------------------------------------
def mp4_to_detail_csv(mp4_name: str) -> str:
    """Strip '_cam#' and change .mp4 → .csv."""
    stem = re.sub(r"_cam\d+", "", mp4_name)
    stem = os.path.splitext(stem)[0]
    return f"{stem}.csv"

def parse_components(cell) -> Set[str]:          # <-- use Set from typing
    """
    Turn a cell like 'A;B;C' into a set of labels.
    Accepts NaN and returns an empty set.
    """
    if pd.isna(cell):
        return set()
    cell = str(cell).strip().strip("[](){}")
    parts = re.split(r"[;,]", cell)
    return {p.strip().lower() for p in parts if p.strip()}

# -------------------------------------------------------------------
# 3.  counters
# -------------------------------------------------------------------
segments       = [f"seg{i}_score" for i in range(1, 5)]
checked_cols   = [f"seg{i}_checked"   for i in range(1, 5)]
unchecked_cols = [f"seg{i}_unchecked" for i in range(1, 5)]

agree_rating = defaultdict(lambda: defaultdict(int))
total_rating = defaultdict(lambda: defaultdict(int))

agree_seg_overall = defaultdict(int)
total_seg_overall = defaultdict(int)

agree_task = defaultdict(int)
total_task = defaultdict(int)

agree_comp = defaultdict(int)
total_comp = defaultdict(int)

missing = []

# -------------------------------------------------------------------
# 4.  iterate rows
# -------------------------------------------------------------------
summary_df = pd.read_csv(MAIN_CSV)

for _, row in summary_df.iterrows():
    csv_name = mp4_to_detail_csv(row["Top"])
    trial_path = os.path.join(DETAIL_DIR, csv_name)
    if not os.path.isfile(trial_path):
        missing.append(csv_name)
        continue

    try:
        trial_df = pd.read_csv(trial_path)
    except Exception:
        continue

    trial_row = trial_df.iloc[0]

    # (a) task_score
    try:
        s_task = int(row["task_score"])
        d_task = int(trial_row["task_score"])
        total_task[s_task] += 1
        if s_task == d_task:
            agree_task[s_task] += 1
        else:                                      ### NEW
            diff_task[s_task - d_task] += 1        ### NEW            
    except Exception:
        pass

    # (b) segment scores
    for seg in segments:
        try:
            s = int(row[seg]); d = int(trial_row[seg])
        except Exception:
            continue
        total_rating[seg][s] += 1
        if s == d:
            agree_rating[seg][s] += 1
            agree_seg_overall[seg] += 1
        else:                                      ### NEW
            diff_seg[seg][s - d] += 1              ### NEW            
        total_seg_overall[seg] += 1

    # (c) component status
    for idx in range(4):
        seg_key = f"seg{idx+1}"

        s_checked = parse_components(row[checked_cols[idx]])
        s_uncheck = parse_components(row[unchecked_cols[idx]])
        d_checked = parse_components(trial_row[checked_cols[idx]])
        d_uncheck = parse_components(trial_row[unchecked_cols[idx]])

        components = s_checked | s_uncheck | d_checked | d_uncheck

        for comp in components:
            s_status = "checked" if comp in s_checked else "unchecked"
            d_status = "checked" if comp in d_checked else "unchecked"

            total_comp[seg_key] += 1
            if s_status == d_status:
                agree_comp[seg_key] += 1

# -------------------------------------------------------------------
# 5.  reporting
# -------------------------------------------------------------------
pct = lambda a, t: (a / t * 100) if t else 0.0

print("\n=== (1) PER-SEGMENT SCORE AGREEMENT BY RATING ===")
hdr = "Segment       " + "  ".join(f"{r:^12}" for r in range(4))
print(hdr); print("-" * len(hdr))
for seg in segments:
    cells = [f"{seg:<13}"]
    for r in range(4):
        a, t = agree_rating[seg][r], total_rating[seg][r]
        cells.append(f"{a}/{t} ({pct(a,t):4.1f}%)")
    print("  ".join(cells))

print("\n=== (2) OVERALL SEGMENT-SCORE AGREEMENT ===")
for seg in segments:
    a, t = agree_seg_overall[seg], total_seg_overall[seg]
    print(f" {seg}: {a}/{t}  →  {pct(a,t):5.1f}%")

print("\n=== (3) COMPONENT STATUS AGREEMENT ===")
for seg_key in [f"seg{i}" for i in range(1, 5)]:
    a, t = agree_comp[seg_key], total_comp[seg_key]
    print(f" {seg_key}: {a}/{t}  →  {pct(a,t):5.1f}%")

print("\n=== (4) TASK-SCORE AGREEMENT ===")
for r in range(4):
    a, t = agree_task[r], total_task[r]
    print(f" Rating {r}: {a}/{t}  →  {pct(a,t):4.1f}%")

if missing:
    print("\nMissing per-trial CSVs:")
    for m in missing:
        print(" •", m)
# ===============================================================
# EXTRA: compute time-per-file and plot
# ===============================================================
from datetime import datetime, timezone
import matplotlib.pyplot as plt

def parse_iso_z(ts: str) -> datetime:
    """
    Parse '2025-05-29T18:38:01.470Z' → timezone-aware datetime.
    Works for .%f (milliseconds) or without.
    """
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"    # replace 'Z' with '+00:00'
    return datetime.fromisoformat(ts)

durations = []        # seconds for each file
csv_files = sorted(f for f in os.listdir(DETAIL_DIR) if f.lower().endswith(".csv"))

for f in csv_files:
    try:
        df = pd.read_csv(os.path.join(DETAIL_DIR, f))
        st = parse_iso_z(df.loc[0, "start_time"])
        et = parse_iso_z(df.loc[0, "end_time"])
        durations.append((et - st).total_seconds())
    except Exception as e:
        print("Skipping", f, ":", e)

if durations:
    avg_sec = sum(durations) / len(durations)
    print(f"\nAverage time per file: {avg_sec/60:.2f} minutes ({avg_sec:.1f} seconds)")

    # ---------- plot ----------
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(durations) + 1), durations, marker="o", linestyle="-")
    plt.xlabel("File index (1–N)")
    plt.ylabel("Duration (seconds)")
    plt.title("Processing Time per Trial File")
    plt.tight_layout()
    plt.show()
else:
    print("No durations computed – check start_time / end_time columns.")

# ------------------------------------------------------------
# (5)  DIRECTIONAL DIFFERENCES  (therapist – per-trial)
# ------------------------------------------------------------
print("\n=== DIRECTIONAL DIFFERENCES: task_score ===")
for delta in range(-3, 4):          # -3 … +3
    if delta == 0: continue
    print(f" Δ {delta:+}: {diff_task[delta]}")

print("\n=== DIRECTIONAL DIFFERENCES: segment scores ===")
for seg in segments:
    print(f"\n {seg}")
    for delta in range(-3, 4):
        if delta == 0: continue
        print(f"   Δ {delta:+}: {diff_seg[seg][delta]}")
