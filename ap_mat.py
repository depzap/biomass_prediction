# ap_mat.py  — safe evaluator for Windows
import os
import multiprocessing
import warnings

# reduce thread pools that sometimes break dataloader workers on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog

# ----- EDIT THESE PATHS -----
VAL_JSON   = r"C:\biomass\labeled_images\result_fixed_file_names_4cls.json"
IMAGES_DIR = r"C:\biomass\images"
WEIGHTS    = r"C:\biomass\output\model_final.pth"   # your trained model file or checkpoint URL
OUTPUT_DIR = r"C:\biomass\output\eval"
NUM_CLASSES = 4
DATASET_NAME = "pasture_val"
# ---------------------------

def prepare_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    # Force single-process dataloader on Windows to avoid worker spawn issues
    cfg.DATALOADER.NUM_WORKERS = 0
    # Use GPU if available
    cfg.MODEL.DEVICE = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    cfg.DATASETS.TEST = (DATASET_NAME,)
    return cfg

def register_dataset():
    # Only register inside main process to avoid repeated registers on worker import
    if DATASET_NAME not in DatasetCatalog.list():
        register_coco_instances(DATASET_NAME, {}, VAL_JSON, IMAGES_DIR)
        print(f"Registered {DATASET_NAME} -> {VAL_JSON} with images in {IMAGES_DIR}")
    else:
        print(f"{DATASET_NAME} already registered (skipping)")

def run_evaluation():
    cfg = prepare_cfg()

    # sanity: print dataset meta
    meta = MetadataCatalog.get(DATASET_NAME)
    print("metadata.json_file:", getattr(meta, "json_file", None))
    print("metadata.thing_classes:", getattr(meta, "thing_classes", None))

    # ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(DATASET_NAME, cfg, False, output_dir=OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, DATASET_NAME)

    print("Running evaluation ... (this will iterate the dataloader)")
    try:
        res = inference_on_dataset(predictor.model, val_loader, evaluator)
        print("Raw eval results:\n", res)
    except RuntimeError as e:
        # Common result: dataloader worker exited unexpectedly
        print("\nRuntimeError during inference_on_dataset:")
        print(e)
        print("\nHints to debug:")
        print("  - We set cfg.DATALOADER.NUM_WORKERS = 0 to avoid spawn issues; confirm the script you run")
        print("  - Ensure VAL_JSON and IMAGES_DIR paths are correct and all referenced images exist.")
        print("  - If images are missing, you will see FileNotFoundError earlier.")
        print("  - Try running with DEVICE='cpu' to see clearer stack traces (set cfg.MODEL.DEVICE = 'cpu')\n")
        raise

if __name__ == "__main__":
    # Required on Windows when spawning subprocesses
    multiprocessing.freeze_support()
    # Register and run evaluation only in main process
    register_dataset()
    run_evaluation()