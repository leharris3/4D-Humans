import torch
import argparse
import time
import os
import cv2
import numpy as np
import logging
import json
import orjson
import lz4.frame
import torch.multiprocessing as mp
import joblib
import msgpack
import gzip

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from glob import glob
from pathlib import Path

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset

# torch optimizations
torch.set_float32_matmul_precision('high')
torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
ANNOTATIONS_DIR = "/mnt/mir/levlevi/nba-plus-statvu-dataset/filtered-clip-annotations"
OUT_DIR = "/mnt/mir/levlevi/nba-plus-statvu-dataset/filtered-clip-3d-poses-hmr-2.0"
NUM_GPUS = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def reduce_precision(input_structure, decimal_places=2):
    """
    Round all values in the input structure to 2 decimal places recursively.
    """
    
    if isinstance(input_structure, dict):
        return {key: reduce_precision(value, decimal_places) for key, value in input_structure.items()}
    elif isinstance(input_structure, list):
        # Recursively apply the function to each item in the list
        return [reduce_precision(item, decimal_places) for item in input_structure]
    elif isinstance(input_structure, (int, float)):
        # Round the number if it's an int or float
        return round(input_structure, decimal_places)
    else:
        # If it's neither a dict, list, int, nor float, return it as is
        return input_structure


def reduce_precision_stack(data, precision=2):
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, (dict, list)):
                    stack.append(v)
                elif isinstance(v, float):
                    item[k] = round(v, precision)
        elif isinstance(item, list):
            for i in range(len(item)):
                if isinstance(item[i], (dict, list)):
                    stack.append(item[i])
                elif isinstance(item[i], float):
                    item[i] = round(item[i], precision)
    return data


def convert_numpy_to_serializable(obj):
    """Convert NumPy objects to JSON-serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert NumPy scalars to native Python types
    return obj  # Return the object if it's not a NumPy type


def default(data):
    """Recursively traverse the dictionary and convert NumPy objects to JSON-serializable formats."""
    if isinstance(data, dict):
        return {key: default(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [default(item) for item in data]
    else:
        return convert_numpy_to_serializable(data)


def convert_tensors_to_numpy_to_list(data):
    """
    Recursively convert all tensor objects in a dictionary and its sub-dictionaries to numpy arrays on the CPU.

    Args:
    data (dict): The dictionary to search through.

    Returns:
    dict: The modified dictionary with tensor objects converted to numpy arrays.
    """
    
    if isinstance(data, dict):
        return {key: convert_tensors_to_numpy_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_tensors_to_numpy_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_tensors_to_numpy_to_list(item) for item in data)
    elif torch.is_tensor(data):
        return data.cpu().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def format_bbxs(bbxs):
    arr = []
    for bbx in bbxs:
        x, y, width, height = bbx['x'], bbx['y'], bbx['width'], bbx['height']
        arr.append([x, y, x+width, y+height])
    return np.array(arr)


def process_frame(frame_idx, frame, annotation, model_cfg, batch_size, device, model):
    """
    Generate 3D-pose data for a single frame.
    """
    
    try:
        frame_obj = annotation['frames'][frame_idx]
        results = {}
        if "bbox" in frame_obj:
            bbxs = frame_obj['bbox']
            bbxs_formatted = format_bbxs(bbxs)
            dataset = ViTDetDataset(model_cfg, frame, bbxs_formatted)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            load_start = time.time()
            for batch in dataloader:
                # logger.info(f"batch loaded in {time.time() - load_start}")
                batch = recursive_to(batch, device)
                forward_start = time.time()
                with torch.no_grad():
                    out = model(batch)
                # logger.info(f"forward pass computed in {time.time() - forward_start}")
                results[str(frame_idx)] = out
                load_start = time.time()
        return results
    except Exception as e:
        logger.info(f"Error processing frame: {frame_idx}")
        logger.info(e)
        return None


@torch.autocast(device_type='cuda', dtype=torch.float16)
def process_clip(model_cfg, model, device, annotation_fp:str):
    """
    Process a single clip and save the results.
    """
    
    annotation = {}
    try:
        with open(annotation_fp, 'r') as f:
            annotation = json.load(f)
    except Exception as e:
        logger.info(f"Failed to load annotation from {annotation_fp}")
        logger.info(e)
        return
    
    # get path to video
    video_fp = annotation_fp.replace("filtered-clip-annotations", "filtered-clips").replace(".json", ".mp4").replace("_annotation", "")
    try:
        assert os.path.exists(video_fp), f"{video_fp} does not exist"
    except Exception as e:
        logger.info(e)
        return
    
    out_fp = annotation_fp.replace("filtered-clip-annotations", "filtered-clip-3d-poses-hmr-2.0").replace("/sun", "/mir")
    compressed_fp = out_fp.replace(".json", "_bin.lz4")
    
    # if compressed_fp exists, skip
    if os.path.isfile(compressed_fp):
        return
    
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    
    # pre-load frames
    results = {}
    frames = []
    video_capture = cv2.VideoCapture(video_fp)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.array(frames)
    
    # predict 3D poses for each bbx in a clip
    # how can we optimize this infer loop?
    # 1. dataloading: can load all bbxs into shared mem up front
    # 2. batch processing in the forward pass
    
    batch_size = 10
    with ThreadPoolExecutor(max_workers=300) as executor:
        future_to_frame = {executor.submit(process_frame, frame_idx, frame, annotation, model_cfg, batch_size, device, model): frame_idx for frame_idx, frame in enumerate(frames)}
        for future in tqdm(as_completed(future_to_frame), desc=f"Processing Video: {video_fp}", total=len(frames)):
            frame_idx = future_to_frame[future]
            result = future.result()
            try:
                if result is not None:
                    results[str(frame_idx)] = result
            except Exception as e:
                logger.info(f"Error adding frame to result: {e}")
            
    # https://stackoverflow.com/questions/68303551/typeerror-object-of-type-tensor-is-not-json-serializable-dict-to-json-error-i 
    
    start_rp = time.time()
    
    # reduce precision to 2-decimal places
    data_json_reduced_precision = reduce_precision(results)
    logger.info(f"Reduced precision in {time.time() - start_rp}")
    
    data_json_reduced_precision = convert_tensors_to_numpy_to_list(data_json_reduced_precision)
    
    # convert json -> bin
    start_pack = time.time()
    packed_data = msgpack.packb(data_json_reduced_precision)
    logger.info(f"Packed data in {time.time() - start_pack}")
    
    # compress data
    start_comp = time.time()
    compressed_packed_data = lz4.frame.compress(packed_data)
    logger.info(f"Compressed data in {time.time() - start_comp}")
    
    # compress and write results
    start = time.time()
    with lz4.frame.open(compressed_fp, 'wb') as f:
        
        # orjson.dumps(results, default=NumpyEncoder)
        # https://pypi.org/project/orjson/2.0.0/
        # data = orjson.dumps(results, default=default)
        # load json obj.
        # data_json = orjson.loads(data)
        
        f.write(compressed_packed_data)
    
    logger.info(f"Wrote results to out in {time.time() - start}")
    
    # remove some exisiting files
    # if os.path.exists(out_fp):
    #     os.remove(out_fp)
            
            
def worker(device, args, file_chunks):
    """
    Process a subset of annotations on `device`.
    """
    
    torch.cuda.set_device(device)
    model, model_cfg = load_hmr2(args.checkpoint)
    model = model.to("cuda")
    # model = torch.compile(model)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    annotation_file_paths = file_chunks[int(device)]
    logger.info(f"Processing {len(annotation_file_paths)} annotations on device {device}")
    for ann_fp in annotation_file_paths:
        try:
            process_clip(model_cfg, model, device, ann_fp, )
        except Exception as e:
            logger.info(e)
            logger.info(f"Failed to process {ann_fp}")
    
    
def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    args = parser.parse_args()
    
    # download_models(CACHE_DIR_4DHUMANS)
    annotation_file_paths = glob(ANNOTATIONS_DIR + "/*/*/*.json")
    file_chunks = [annotation_file_paths[i::NUM_GPUS] for i in range(NUM_GPUS)]
    mp.spawn(
        worker,
        args=(args, file_chunks),
        nprocs=NUM_GPUS,
        join=True,
    )
    
    
if __name__ == '__main__':
    main()
