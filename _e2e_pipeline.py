import torch
import argparse
import os
import cv2
import msgpack
import lz4.frame
import numpy as np
import logging
import shutil
import random
import warnings
import torch.multiprocessing as mp

from glob import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from _pipeline import format_bbxs, reduce_precision, convert_tensors_to_numpy_to_list
from _helpers import ClipAnnotation

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
BATCH_SIZE = 32
NUM_GPUS = 8
ANNOTATIONS_DIR = "/mnt/mir/levlevi/nba-plus-statvu-dataset/filtered-clip-annotations"

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

@torch.autocast(device_type="cuda", dtype=torch.float16)
@torch.inference_mode()
def process_frame(frame_idx, img_path, detector, model_cfg, model, device):

    img_cv2 = cv2.imread(str(img_path))
    # Detect humans in image
    # TODO: we can use pre-extracted detections
    det_out = detector(img_cv2)

    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    # Run HMR2.0 on all detected humans
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            hmr_out = model(batch)
            del hmr_out['pred_smpl_params']
            del hmr_out['focal_length']
            del hmr_out['pred_keypoints_3d']
            del hmr_out['pred_keypoints_2d']
        return boxes, hmr_out


@torch.autocast(device_type="cuda", dtype=torch.float16)
@torch.inference_mode()
def process_ann(model_cfg, model, detector, device, annotations_fp: str):

    ann = ClipAnnotation(annotations_fp)
    if ann.video_fp is None:
        print(f"Skipping: {annotations_fp}")
        return
    tmp_frames_dir = f"/playpen-storage/levlevi/tmp/{ann.basename}"
    os.makedirs(tmp_frames_dir, exist_ok=True)

    ann.save_fames(tmp_frames_dir)

    # TODO: this completely DESTROYS model output quality
    # model = torch.compile(model)
    model.eval()

    out_fp = ann.three_d_poses_fp
    if os.path.isfile(out_fp):
        logger.info(f"Already processed 3d-pose data for file: {out_fp}... skipping!")
        return
    
    # make output directory if it does not exist
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    # Get all demo images that end with .jpg or .png
    img_paths = glob(tmp_frames_dir + "/*.png")
    results = {}

    # process all imgs in a dir
    # OOM errors w/ bs >= 24
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(
                process_frame,
                idx,
                img_path,
                detector,
                model_cfg,
                model,
                device,
            ): idx
            for idx, img_path in enumerate(img_paths)
        }
        for future in tqdm(
        as_completed(futures),
        total=len(img_paths),
        desc=f"Processing annotation {os.path.basename(annotations_fp).replace('.json', '')}"
        ):
            frame_idx = futures[future]
            boxes, hmr_out = future.result()
            if boxes is not None and hmr_out is not None:
                results[str(frame_idx)] = {}
                results[str(frame_idx)]["boxes"] = boxes
                results[str(frame_idx)]["hmr_out"] = hmr_out
            else:
                logger.error(f"Boxes or HMR outputs are `None`")

    # convert all data -> list
    results = convert_tensors_to_numpy_to_list(
        results
    )
    
    # reduce precision to 2-decimal places
    data_json_reduced_precision = reduce_precision(results)
    
    # convert json -> bin
    packed_data = msgpack.packb(data_json_reduced_precision)
    
    # compress data
    compressed_packed_data = lz4.frame.compress(packed_data)
    
    # write results
    with lz4.frame.open(out_fp, "wb") as f:
        # orjson.dumps(results, default=NumpyEncoder)
        # https://pypi.org/project/orjson/2.0.0/
        # data = orjson.dumps(results, default=default)
        # load json obj.
        # data_json = orjson.loads(data)
        f.write(compressed_packed_data)

    # remove the tmp frames dir
    shutil.rmtree(tmp_frames_dir)


def worker(device, args, file_chunks):
    """
    Process a subset of annotations on `device`.
    """

    torch.cuda.set_device(device)
    model, model_cfg = load_hmr2(args.checkpoint)
    model = model.to("cuda")
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

    if args.detector == "vitdet":
        from detectron2.config import LazyConfig
        import hmr2

        cfg_path = (
            Path(hmr2.__file__).parent
            / "configs"
            / "cascade_mask_rcnn_vitdet_h_75ep.py"
        )
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.detector == "regnety":
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    annotation_file_paths = file_chunks[int(device)]
    logger.info(
        f"Processing {len(annotation_file_paths)} annotations on device {device}"
    )
    for ann_fp in annotation_file_paths:
        try:
            process_ann(model_cfg, model, detector, device, ann_fp)
        except Exception as e:
            logger.error(f"Error processing annotation: {ann_fp}")
            logger.error(f"e: {e}")
        

def main():
    parser = argparse.ArgumentParser(description="HMR2 demo code")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        default="example_data/images",
        help="Folder with input images",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="demo_out",
        help="Output folder to save rendered results",
    )
    parser.add_argument(
        "--side_view",
        dest="side_view",
        action="store_true",
        default=False,
        help="If set, render side view also",
    )
    parser.add_argument(
        "--top_view",
        dest="top_view",
        action="store_true",
        default=False,
        help="If set, render top view also",
    )
    parser.add_argument(
        "--full_frame",
        dest="full_frame",
        action="store_true",
        default=False,
        help="If set, render all people together also",
    )
    parser.add_argument(
        "--save_mesh",
        dest="save_mesh",
        action="store_true",
        default=False,
        help="If set, save meshes to disk also",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="vitdet",
        choices=["vitdet", "regnety"],
        help="Using regnety improves runtime",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference/fitting"
    )
    parser.add_argument(
        "--file_type",
        nargs="+",
        default=["*.jpg", "*.png"],
        help="List of file extensions to consider",
    )
    args = parser.parse_args()

    # download_models(CACHE_DIR_4DHUMANS)
    annotation_file_paths = glob(ANNOTATIONS_DIR + "/*/*/*.json")
    
    # shuffle paths for concurrent processing on multiple nodes
    random.shuffle(annotation_file_paths)
    
    file_chunks = [annotation_file_paths[i::NUM_GPUS] for i in range(NUM_GPUS)]
    mp.spawn(
        worker,
        args=(args, file_chunks),
        nprocs=NUM_GPUS,
        join=True,
    )


if __name__ == "__main__":
    main()
