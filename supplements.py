import logging
import open3d as o3d
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
from azure.storage.blob import BlobServiceClient, AccessPolicy, ContainerSasPermissions, PublicAccess


# Azure Storage Account: pointclouds1
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING_2')
# Create the BlobServiceClient object which will be used to create a container client.
blob_service_client_2 = BlobServiceClient.from_connection_string(connect_str)


class PcdOps:

    """class to implement poisson reconstruction and outlier removal"""

    def __init__(self, output_dir, container_name):
        self.container_name = container_name
        self.output_dir = output_dir
        self.input_dense = output_dir + '/dense.ply'
        self.output_inlier_cloud = output_dir + '/dense_inlier.ply'
        self.output_poisson = output_dir + '/poisson.ply'

    def fetch_file_from_remote(self, file):

        container_client = blob_service_client_2.get_container_client(self.container_name)
        blob_client = container_client.get_blob_client(file)  # Instantiate BlobClient

        try:
            with open(os.path.join(self.output_dir, file), "wb") as my_blob:
                    download_stream = blob_client.download_blob()
                    my_blob.write(download_stream.readall())

        except Exception:
            return

    def reconstruct(self, outliers=False, poisson=False):

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

