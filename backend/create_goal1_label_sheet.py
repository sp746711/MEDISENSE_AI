import pandas as pd
from pathlib import Path

source = Path("xray_goal1_unique_nonplaceholder_audit.csv")
if not source.exists():
    raise SystemExit(f"Missing audit file: {source.resolve()}")

df = pd.read_csv(source)

if "sha256" not in df.columns:
    raise SystemExit("CSV does not contain the required sha256 column.")

unique = df.drop_duplicates(subset=["sha256"]).copy()
unique["verified_anatomy"] = ""
unique["review_notes"] = ""

output = Path("xray_goal1_ground_truth_labels.csv")
unique.to_csv(output, index=False)

print("Original rows:", len(df))
print("Unique image contents:", len(unique))
print("Created:", output.resolve())
print("\nFirst rows:")
print(unique[["filename", "candidate", "confidence",
              "second_candidate", "second_confidence",
              "verified_anatomy", "review_notes"]]
      .head(10).to_string(index=False))
