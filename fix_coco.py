import json, os
from pathlib import Path

# CHANGE THESE TO MATCH YOUR SYSTEM
PROJECT = Path(r"C:\biomass\labeled_images")
IMAGES_DIR = Path(r"C:\biomass\images")

# Load original JSON
input_json = PROJECT/"result.json"
cj = json.load(open(input_json, "r", encoding="utf-8"))

print("Fixing file_name entries...", len(cj["images"]))

missing = 0

for im in cj["images"]:
    # Replace with basename only
    original = im["file_name"]
    basename = os.path.basename(original)
    im["file_name"] = basename

    # Count missing files
    if not (IMAGES_DIR / basename).exists():
        missing += 1

print("Missing after rewrite:", missing)

fixed_json = PROJECT/"result_fixed.json"
json.dump(cj, open(fixed_json, "w", encoding="utf-8"), indent=2)
print("Saved:", fixed_json)
