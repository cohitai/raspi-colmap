import docker
import logging
import open3d as o3d
import numpy as np
import os
from pathlib import Path


class DockerizedColmap:
    # container's inner directory structure.
    DATASET_PATH = "/root/data"
    DATABASE_PATH = "/root/data/output/database.db"
    IMAGESET_PATH = "/root/data/dataset-m"
    OUTPUT_PATH = "/root/data/output"
    SPARSE_PATH = "/root/data/output/0"
    NEW_SPARSE_PATH = "/root/data/output/sparse"
    DENSE_PATH = "/root/data/output"
    DENSE_PLY_PATH = "/root/data/output/dense.ply"

    def __init__(self, local_dir, masked_dir, output_dir):

        self.local_dir = local_dir
        self.masked_dir = masked_dir
        self.output_dir = output_dir
        self.input_dense = output_dir + '/dense.ply'
        self.output_inlier_cloud = output_dir + '/dense_inlier.ply'
        self.output_poisson = output_dir + '/poisson.ply'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def reconstruct(self, outliers=False, poisson=False):
        """ COLMAP in docker
           :params: boolean; outliers, adds the outliers feature,
                    boolean; poisson, adds Poisson meshing. """

        client = docker.from_env()  # connect to docker daemon

        mount_dict = {self.local_dir: {'bind': self.IMAGESET_PATH[:-2], 'mode': 'rw'},
                      self.masked_dir: {'bind': self.IMAGESET_PATH, 'mode': 'rw'},
                      self.output_dir: {'bind': self.OUTPUT_PATH, 'mode': 'rw'}}

        cmd1 = f"colmap feature_extractor --database_path {self.DATABASE_PATH} --image_path {self.IMAGESET_PATH}"
        cmd2 = f"colmap exhaustive_matcher \
            --database_path {self.DATABASE_PATH}"
        cmd3 = f"colmap mapper \
            --database_path {self.DATABASE_PATH} \
            --image_path {self.IMAGESET_PATH} \
            --output_path {self.OUTPUT_PATH}"
        cmd4 = f"colmap image_undistorter \
            --image_path {self.IMAGESET_PATH} \
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

        self._run_colmap(client, colmap_cmds, mount_dict, self.DATASET_PATH)

        """outliers removal"""
        if outliers:
            pcd = o3d.io.read_point_cloud(self.input_dense)
            inlier_cloud, outlier_cloud = self._remove_outliers(pcd, self._create_outliers_list(pcd))
            logging.info(f"number of points removed by knn:{len(pcd.points) - len(inlier_cloud.points)}")
            #  save to file
            o3d.io.write_point_cloud(self.output_inlier_cloud, inlier_cloud, write_ascii=True, compressed=False,
                                     print_progress=False)

        """poisson meshing"""
        if poisson:
            pcd = o3d.io.read_point_cloud(self.output_inlier_cloud) if Path(self.output_inlier_cloud).is_file() \
                else o3d.io.read_point_cloud(self.input_dense)
            poisson_mesh = self.poisson_reconstruction(pcd)
            o3d.io.write_triangle_mesh(self.output_poisson, poisson_mesh, write_ascii=True, compressed=False)

    @staticmethod
    def _run_colmap(client, cmd_dict, mount_dict, wd):
        """ dockerized COLMAP"""

        def _execute_colmap_command(cl, cmd, mount_dict, wd, container_name='colmap:test'):
            return cl.containers.run(container_name, cmd, volumes=mount_dict, working_dir=wd, runtime="nvidia",
                                     detach=False, auto_remove=True)

        for colmap_command, command in cmd_dict.items():
            logging.info(f"executing colmap: {colmap_command}")
            _execute_colmap_command(client, command, mount_dict, wd)

    @staticmethod
    def _create_outliers_list(pcd):
        # Convert open3d format to numpy array
        pcd_colors = np.asarray(pcd.colors) * 255
        pcd_colors_summed = np.expand_dims(pcd_colors.sum(axis=1), axis=1)
        return np.where(np.any(pcd_colors_summed > 720, axis=1))[0].tolist()

    @staticmethod
    def _remove_outliers(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind, invert=True)
        outlier_cloud = cloud.select_by_index(ind, invert=False)
        return inlier_cloud.remove_radius_outlier(nb_points=100, radius=0.1)[0], outlier_cloud

    @staticmethod
    def poisson_reconstruction(pcd):
        return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                         depth=12, width=0, scale=1.1,
                                                                         linear_fit=False)[0]
