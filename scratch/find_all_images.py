from pathlib import Path

files = []
for p in Path(".").rglob("*"):
    if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        files.append(p)

print(f"Total images found: {len(files)}")
for f in files:
    name_l = f.name.lower()
    if any(k in name_l for k in ["foot", "skull", "cervical", "spine", "elbow", "ankle", "forearm", "hand", "shoulder", "knee", "abdomen"]):
        print(f"  {f}")
