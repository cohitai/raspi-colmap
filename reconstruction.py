import docker
import logging
import open3d as o3d
import numpy as np
import os
from pathlib import Path


class DockerizedColmap:

    # container image:
    # CONTAINER_IMAGE = "colmap/colmap:latest"

    # container's inner directory structure.
    DATASET_PATH = "/root/data"
    DATABASE_PATH = "/root/data/output/database.db"
    IMAGE_PATH = "/root/data/dataset-m"
    OUTPUT_PATH = "/root/data/output"
    SPARSE_PATH = "/root/data/output/0"
    NEW_SPARSE_PATH = "/root/data/output/sparse"
    DENSE_PATH = "/root/data/output"
    DENSE_PLY_PATH = "/root/data/output/dense.ply"

    def __init__(self, local_dir, masked_dir, output_dir, container_image):

        self.local_dir = local_dir
        self.masked_dir = masked_dir
        self.output_dir = output_dir
        self.input_dense = output_dir + '/dense.ply'
        self.container_image = container_image

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def reconstruct(self):
        """ COLMAP in docker """

        client = docker.from_env()  # connect to docker daemon

        max_num_matches = 132768

        mount_dict = {self.local_dir: {'bind': self.IMAGE_PATH[:-2], 'mode': 'rw'},
                      self.masked_dir: {'bind': self.IMAGE_PATH, 'mode': 'rw'},
                      self.output_dir: {'bind': self.OUTPUT_PATH, 'mode': 'rw'}}

        cmd1 = f"colmap feature_extractor --database_path {self.DATABASE_PATH} --image_path {self.IMAGE_PATH}"

        feature_extractor_cmd = f'''
                colmap feature_extractor \
                    --database_path={self.DATABASE_PATH} \
                    --image_path={self.IMAGE_PATH} \
                    --ImageReader.single_camera=1 \
                    --ImageReader.default_focal_length_factor=0.69388 \
                    --SiftExtraction.peak_threshold=0.004 \
                    --SiftExtraction.max_num_features=8192 \
                    --SiftExtraction.edge_threshold=16 \
                    --ImageReader.camera_model=SIMPLE_PINHOLE '''

        cmd2 = f"colmap exhaustive_matcher \
            --database_path {self.DATABASE_PATH}"

        exhaustive_matcher_cmd = f'''colmap exhaustive_matcher \
                        --database_path={self.DATABASE_PATH} \
                        --SiftMatching.multiple_models=0 \
                        --SiftMatching.max_ratio=0.8 \
                        --SiftMatching.max_error=4.0 \
                        --SiftMatching.max_distance=0.7 \
                        --SiftMatching.max_num_matches={max_num_matches}'''

        cmd3 = f"colmap mapper \
            --database_path {self.DATABASE_PATH} \
            --image_path {self.IMAGE_PATH} \
            --output_path {self.OUTPUT_PATH}"

        mapper_cmd = f'''
                colmap mapper \
                    --database_path={self.DATABASE_PATH} \
                    --image_path={self.IMAGE_PATH} \
                    --output_path={self.SPARSE_PATH}'''

        cmd4 = f"colmap image_undistorter \
            --image_path {self.IMAGE_PATH} \
            --input_path {self.SPARSE_PATH} \
            --output_path {self.DENSE_PATH} \
            --output_type COLMAP"

        cmd5 = f"colmap patch_match_stereo \
            --workspace_path {self.DENSE_PATH} \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true"
        cmd6 = f"colmap stereo_fusion \
            --workspace_path {self.DENSE_PATH} \
            --workspace_format COLMAP \
            --input_type geometric \
            --output_path {self.DENSE_PLY_PATH}"
        cmd7 = f"colmap model_converter \
                    --input_path {self.SPARSE_PATH} \
                    --output_path {self.SPARSE_PATH} \
                    --output_type TXT"

        colmap_cmds = {"feature_extractor": cmd1, "exhaustive_matcher": cmd2, "mapper": cmd3, "image_undistorter": cmd4,
                       "patch_match_stereo": cmd5, "stereo_fusion": cmd6, "model_converter": cmd7}

        self._run_colmap(client, colmap_cmds, mount_dict, self.DATASET_PATH, self.container_image)

    @staticmethod
    def _run_colmap(client, cmd_dict, mount_dict, wd, container_image):
        """ dockerized COLMAP"""

        def _execute_colmap_command(cl, cmd, mount_dict, wd, container_image=container_image):
            return cl.containers.run(container_image, cmd, volumes=mount_dict, working_dir=wd, runtime="nvidia",
                                     detach=False, auto_remove=True)

        for colmap_command, command in cmd_dict.items():
            logging.info(f"executing colmap: {colmap_command}")
            _execute_colmap_command(client, command, mount_dict, wd)
