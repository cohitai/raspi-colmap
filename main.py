import os
import shutil
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from datetime import datetime
import logging
import sys
import glob
import numpy as np
import cv2


# create logger
logging.getLogger('raspi-colmap')
logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.DEBUG)

# Azure connecting info. added as environment variable

os.environ['AZURE_STORAGE_CONNECTION_STRING'] = 'DefaultEndpointsProtocol=https;AccountName=blobsdb;AccountKey=tJK43kihAcaeZMjcegWFcyg8tsFmOr9f2Kn8q6NUinVSJW5O3jymYbjaiGBjmx8Ibq5LsBVPcABvYeV+tUCPnQ==;EndpointSuffix=core.windows.net'
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
# Create the BlobServiceClient object which will be used to create a container client.
blob_service_client = BlobServiceClient.from_connection_string(connect_str)


class MaskAzure:

    local_dir = './d1'
    target_dir = './d2'

    # def __init__(self):

    def init(self):
        if not os.path.exists(self.target_dir): os.makedirs(self.target_dir)
        if not os.path.exists(self.local_dir): os.makedirs(self.local_dir)

    def delete(self):
        shutil.rmtree(self.local_dir)  # delete directory d1
        shutil.rmtree(self.target_dir)  # delete directory d2

    @staticmethod
    def retrieve_last_k_containers(k):

        """function retrieves last k containers from Azure Blobs Storage"""

        container_list = []
        for i, item in enumerate(reversed(list(blob_service_client.list_containers()))):
            if i == k:
                break
            timestamp = datetime.fromtimestamp(int(item["name"].split("-")[0]))
            logging.debug(f"container downloaded with timestamp: {timestamp} ")
            container_list.append(item["name"])
        return container_list

    def dl_blobs_to_local(self, container_list):

        def _save_blob(blob_name, container_path, client):

            """function downloads and saves a single blob"""

            with open(container_path + "/" + str(blob_name), "wb") as my_blob:
                blob_client = client.get_blob_client(blob_name)  # Instantiate a new BlobClient
                download_stream = blob_client.download_blob()
                my_blob.write(download_stream.readall())

        for container in container_list:
            dir_name = self.local_dir + "/" + container
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                continue

            container_client = blob_service_client.get_container_client(container)
            for blob in container_client.list_blobs():
                _save_blob(blob.name, dir_name, container_client)

    @staticmethod
    def organize_container(container_path):

        """ :recieves: containers path,
        :returns: dictionary, keys are camerasIds, values are path_to_image´s. """

        files = [file.split("/")[-1] for file in glob.glob(container_path + "/*")]
        cameras = list(set([file[0:2] for file in files]))
        blob2crop_dictionary = {}

        # dict init
        for cam in cameras:
            blob2crop_dictionary[cam] = []
        for file in files:
            blob2crop_dictionary[file[0:2]].append(container_path + "/" + file)
        for cam in cameras:
            blob2crop_dictionary[cam] = sorted(blob2crop_dictionary[cam])

        return blob2crop_dictionary

    def crop_session(self, blob2crop):
        """computes crop parameters for blobs in blob2crop """

        def _compute_diffs(im1, im2, im3):
            return [cv2.subtract(im2, im1), cv2.subtract(im3, im2)]

        def _find_crop_indices(img_diff, axis=0):

            def _derivative(series, interval=1):
                """derivative auxiliary"""
                diff = list()
                for i in range(len(series)):
                    value = series[i] - series[i - interval]
                    diff.append(value)
                return np.array(diff)

            def _find_turning_pts(np_array, interval):
                """function searches for left/right turning/jumping points in <means_array>,
                    :param:
                    interval is an integer, representing a sequel of zeros which comes before/after the suspected
                    point"""
                left_turn = []
                right_turn = []

                for t in range(interval, len(np_array)):
                    if np_array[t] != 0:
                        for j in range(1, interval + 1):
                            if np_array[t - j] != 0:
                                break
                        if j == interval:
                            left_turn.append(t)

                for t in range(0, len(np_array) - interval):
                    if np_array[t] != 0:
                        for j in range(1, interval + 1):
                            if np_array[t + j] != 0:
                                break
                        if j == interval:
                            right_turn.append(t)

                return [0] + left_turn + [len(np_array)], [0] + right_turn + [len(np_array)]

            # reduce img to vertical mean vector
            means_array = img_diff.mean(axis=axis)

            # compute median for clipping
            means_array_median = (means_array.max() - means_array.min()) / 2

            means_array_threshold = means_array_median / 2 if axis == 0 else 1.1 * means_array_median

            # aux clipper
            clip = lambda t: 0 if t - means_array_threshold < 0 else t

            # apply on mean vec on M
            means_array_mod = np.array([clip(t) for t in means_array])

            # find suspected points for the main objects' boundaries
            right, left = _find_turning_pts(abs(_derivative(means_array_mod)), 50)

            # compute the biggest component
            ind_max = np.argmax(list(map(int.__sub__, left, right)))

            # safety interval to be added
            means_array_safe = 100 if axis == 0 else 150
            crop_interval = (
            max(0, right[ind_max] - means_array_safe), min(len(means_array_mod), left[ind_max] + means_array_safe))

            return crop_interval

        cameras = list(blob2crop.keys())
        result_dict = {}

        for cam in cameras:
            grey = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in d[cam]]
            img_inds = [x for x in range(len(grey))]

            for ind in img_inds:
                diff_ind = list(np.argsort([abs(x - ind) for x in [x for x in range(len(grey))]])[:3])
                d_im1, d_im2 = _compute_diffs(grey[diff_ind[1]], grey[diff_ind[0]], grey[diff_ind[2]])

                hl1, hr1 = _find_crop_indices(d_im1, axis=0)
                hl2, hr2 = _find_crop_indices(d_im2, axis=0)
                vl1, vr1 = _find_crop_indices(d_im1, axis=1)
                vl2, vr2 = _find_crop_indices(d_im2, axis=1)

                hlc, hrc = min(hl1, hl2), max(hr1, hr2)
                vlc, vrc = min(vl1, vl2), max(vr1, vr2)

                result_dict[d[cam][ind]] = (hlc, hrc, vlc, vrc)

        return result_dict






#MaskAzure().init()
#containers = MaskAzure().retrieve_last_k_containers(1)
#MaskAzure().dl_blobs_to_local(containers)
#MaskAzure().delete()
d = MaskAzure().organize_container("/home/liteandfog/raspi-colmap/d1/1639055186-836341")
print(MaskAzure().crop_session(d))

