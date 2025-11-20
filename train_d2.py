# train_d2.py  (Option A — 4 classes, cleaned JSON)
import os
import multiprocessing

from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo

# ---- paths (change if needed) ----
TRAIN_JSON = r"C:\biomass\labeled_images\result_fixed_file_names_4cls.json"
IMAGES_DIR  = r"C:\biomass\images"
DATASET_NAME = "pasture_train_v2"

# Register dataset (safe guard: catch re-registration errors)
try:
    register_coco_instances(DATASET_NAME, {}, TRAIN_JSON, IMAGES_DIR)
    print(f"Registered {DATASET_NAME} -> {TRAIN_JSON} with images in {IMAGES_DIR}")
except Exception as e:
    # Often fails when the dataset was registered previously in the same process.
    # This is not fatal if the metadata is already correct — restart Python if you changed JSON or classes.
    print("Dataset registration warning (may already be registered):", e)

# --- Config setup ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Use COCO weights as backbone initialization
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# ---------------- Memory / Windows-safe adjustments ----------------
# 1) Tiny batch size (very important on small GPUs)
cfg.SOLVER.IMS_PER_BATCH = 1   # 1 image per GPU

# 2) Windows: single-process data loading to avoid spawn errors
cfg.DATALOADER.NUM_WORKERS = 0

# 3) Reduce image sizes used in training (save memory)
cfg.INPUT.MIN_SIZE_TRAIN = (512,)    # smaller short edge
cfg.INPUT.MAX_SIZE_TRAIN = 800

# 4) Mask training: keep True if you *need* instance masks. Set False to save a lot of memory.
# If you get CUDA OOM, change this to False.
cfg.MODEL.MASK_ON = True

# 5) Enable AMP (mixed precision) to reduce memory & speed up training (if supported)
cfg.SOLVER.AMP.ENABLED = True

# 6) Learning rate / iterations (tweak as needed)
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000

# 7) Fewer RPN proposals (optional memory/perf trade-off)
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

# 8) Dataset & classes
cfg.DATASETS.TRAIN = (DATASET_NAME,)
cfg.DATASETS.TEST = ()  # no test dataset for now
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # must match number of classes in cleaned JSON

# 9) Output folder
cfg.OUTPUT_DIR = os.path.abspath("./output")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Optional: force CPU for debugging (very slow) - uncomment to use
# cfg.MODEL.DEVICE = "cpu"

# ----------------- quick diagnostic -----------------
meta = None
try:
    from detectron2.data import MetadataCatalog
    meta = MetadataCatalog.get(DATASET_NAME)
    print("Registered classes:", getattr(meta, "thing_classes", None))
    print("metadata.json_file:", getattr(meta, "json_file", None))
except Exception as e:
    print("MetadataCatalog warning:", e)

# Put training inside this guard to avoid multiprocessing spawn errors on Windows
if __name__ == "__main__":
    # enable freeze_support on Windows
    multiprocessing.freeze_support()

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
