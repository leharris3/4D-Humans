import logging
import ujson
import os
import lz4.frame
import msgpack
import cv2

logging.basicConfig(level=logging.INFO)


class ClipAnnotation:
    """
    Each clip in our dataset contains data scattered across many different file and data formats.
    This class is intended to simplify the process of parsing differnt annotations types for a single clip.
    """

    CLIPS_DIR = "filtered-clips"
    ANNOTATIONS_DIR = "filtered-clip-annotations"
    THREE_D_POSES_DIR = "filtered-clip-3d-poses-hmr-2.0"

    def __init__(self, annotation_fp: str):
        """
        Given a path to a primary-annotation file, derive the paths to all other annotations for a given clip.

        Params
        :annotation_fp: a path to a `.json` file containing the primary annotations for each frame in a clip.
        """

        assert os.path.isfile(
            annotation_fp
        ), f"Error: {annotation_fp} is not a valid file"
        try:
            with open(annotation_fp, "r") as f:
                _ = ujson.load(f)
        except Exception as e:
            logging.error(f"Failed to open annotation file at {annotation_fp}")
            raise Exception()

        self.annotations_fp = annotation_fp
        self.basename = (
            os.path.basename(annotation_fp)
            .replace(".json", "")
            .replace("_annotation", "")
        )
        self.video_fp = (
            annotation_fp.replace(
                ClipAnnotation.ANNOTATIONS_DIR, ClipAnnotation.CLIPS_DIR
            )
            .replace("_annotation", "")
            .replace(".json", ".mp4")
        )
        try:
            assert os.path.isfile(self.video_fp)
        except:
            logging.warn(
                f"Clip video file path: {self.video_fp}, does not exist. Setting this attribute to None."
            )
            self.video_fp = None

        self.three_d_poses_fp = annotation_fp.replace(
            ClipAnnotation.ANNOTATIONS_DIR, ClipAnnotation.THREE_D_POSES_DIR
        ).replace(".json", "_bin.lz4")
        try:
            assert os.path.isfile(self.three_d_poses_fp)
        except:
            logging.warning(
                f"3D-pose file path: {self.three_d_poses_fp}, does not exist. Setting this attribute to None."
            )
            # self.three_d_poses_fp = None

    def get_annotations(self):
        with open(self.annotations_fp, "r") as f:
            return ujson.load(f)

    def get_3d_pose_data(self):
        with lz4.frame.open(self.three_d_poses_fp, "rb") as compressed_file:
            # Step 2: Decompress the data
            compressed = compressed_file.read()
            compressed_data = lz4.frame.decompress(compressed)
        # Step 3: Deserialize using msgpack
        decompressed_data = msgpack.unpackb(compressed_data, raw=False)
        # Step 4: Handle any remaining tensor-like structures
        # Assuming that all tensor data was converted to lists, no further action is needed.
        return decompressed_data

    def get_frames(self):
        cap = cv2.VideoCapture(self.video_fp)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames
    
    def save_fames(self, to:str):
        cap = cv2.VideoCapture(self.video_fp)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            out_fp = os.path.join(to, '%.6d' % (idx) + '.png')
            cv2.imwrite(out_fp, frame)
            idx += 1