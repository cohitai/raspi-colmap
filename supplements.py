import logging
import open3d as o3d
from functools import reduce
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

    def compute_volume(self):

        pcd = o3d.io.read_point_cloud(self.output_inlier_cloud)
        mesh = o3d.io.read_triangle_mesh(self.output_poisson)
        #axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd,axes])

        volume = reduce(lambda a, b: a + self.volume_under_triangle(b),
                        self.get_triangles_vertices(mesh.triangles, mesh.vertices), 0)

        surface_area = mesh.get_surface_area()

        #print(pcd.compute_convex_hull())

        print(f"The volume of the mesh is: {round(volume, 4)} m3")
        print(f"The surface area of the mesh is: {surface_area} m2")

    @staticmethod
    def compute_segment_plane(pcd):
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=10000)

        [a, b, c, d] = plane_model
        plane_pcd = pcd.select_by_index(inliers)
        plane_pcd.paint_uniform_color([1.0, 0, 0])
        stockpile_pcd = pcd.select_by_index(inliers, invert=True)
        stockpile_pcd.paint_uniform_color([0, 0, 1.0])

        o3d.visualization.draw_geometries([plane_pcd, stockpile_pcd, axes])

    @staticmethod
    def get_triangles_vertices(triangles, vertices):
        triangles_vertices = []
        for triangle in triangles:
            new_triangles_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]
            triangles_vertices.append(new_triangles_vertices)
        return np.array(triangles_vertices)

    @staticmethod
    def volume_under_triangle(triangle):
        p1, p2, p3 = triangle
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3
        return abs((z1 + z2 + z3) * (x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3) / 6)
