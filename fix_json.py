# fix_json.py
import json
IN_JSON  = r"C:\biomass\labeled_images\result_fixed_file_names.json"
OUT_JSON = r"C:\biomass\labeled_images\result_fixed_file_names_4cls.json"

with open(IN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

keep_cats = [c for c in data.get("categories", []) if c.get("name") != "verified"]
keep_ids = {c["id"] for c in keep_cats}
keep_anns = [a for a in data.get("annotations", []) if a.get("category_id") in keep_ids]

data["categories"]   = keep_cats
data["annotations"]  = keep_anns

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f)
print("Wrote:", OUT_JSON)
