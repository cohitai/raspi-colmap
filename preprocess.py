import os
import shutil
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import logging
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import product
import math
import subprocess
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# create logger
# logging.getLogger('raspi-colmap')
# logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.DEBUG)

SERVER_PWD = bytes(os.getenv('SERVER_PWD'), 'utf-8')

# Azure Storage Account: blobsdb
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING_1')
# Create the BlobServiceClient object which will be used to create a container client.
blob_service_client_1 = BlobServiceClient.from_connection_string(connect_str)


class MaskAzure:

    def __init__(self, local_dir, target_dir, output_dir, supp_dir):
        # directory to store raw containers
        self.local_dir = local_dir
        # directory to store containers after cropping
        self.target_dir = target_dir
        # directory to store colmap output
        self.output_dir = output_dir
        # directory to store the final output
        self.supp_dir = supp_dir

    def init(self):
        """method for initializing the workspace"""
        if os.path.exists(self.local_dir):
            shutil.rmtree(self.local_dir)
        os.makedirs(self.local_dir)

        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir)

        if os.path.exists(self.output_dir):
            subprocess.Popen(['sudo', '-S', 'rm', "-r", self.output_dir], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(input=SERVER_PWD)
        os.makedirs(self.output_dir)

        if os.path.exists(self.supp_dir):
            shutil.rmtree(self.supp_dir)
        os.makedirs(self.supp_dir)

    def fetch_last_container(self, debug_container=None):
        """method to download last container """

        container = debug_container if debug_container else self.retrieve_last_k_containers(1)
        self.dl_blobs_to_local(container)

        return container[0]  # return last container's name

    @staticmethod
    def retrieve_last_k_containers(k):
        """function retrieves last k containers from Azure Blobs Storage."""
        container_list = []
        for i, item in enumerate(reversed(list(blob_service_client_1.list_containers()))):
            if i == k:
                break
            timestamp = datetime.fromtimestamp(int(item["name"].split("-")[0]))
            logging.debug(f"container downloaded with timestamp: {timestamp} ")
            container_list.append(item["name"])
        return container_list

    def dl_blobs_to_local(self, container_list):
        """method downloads containers from [container_list] to local."""

        def _save_blob(blob_name, container_path, client):
            """auxiliary function: downloads <blob_name> and saves at <container_path>."""

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

            container_client = blob_service_client_1.get_container_client(container)
            for blob in container_client.list_blobs():
                _save_blob(blob.name, dir_name, container_client)

    @staticmethod
    def organize_container(container_path):
        """ function creates a dictionary which describes the container structure.
        :param: containers path.
        :returns: a sorted dictionary, keys are camerasIds, values are path_to_image´s. """

        def _retrieve_time(s):
            start = s.find("img") + len("img")
            end = s.find(".jpg")
            return float(s[start:end].replace("-", "."))

        files = [file.split("/")[-1] for file in glob.glob(container_path + "/*")]
        cameras = list(set([file[0:2] for file in files]))
        blob2crop_dictionary = {}

        # dict init
        for cam in cameras:
            blob2crop_dictionary[cam] = []
        for file in files:
            blob2crop_dictionary[file[0:2]].append(container_path + "/" + file)
        for cam in cameras:
            blob2crop_dictionary[cam].sort(key=lambda x: _retrieve_time(x))

        return blob2crop_dictionary

    @staticmethod
    def crop_session(blob2crop):
        """computes crop parameters for blobs in blob2crop.
           :param: blob2crop a dictionary resulted from organize_container (giving info on container and blobs); """

        def _compute_diffs(im1, im2, im3):
            """auxiliary function to subtract images using opencv."""

            return [cv2.subtract(im2, im1), cv2.subtract(im3, im2)]

        def _find_crop_indices(img_diff):
            """auxiliary function to compute crop indices by detecting moving objects in the frames"""

            def _derivative(series, interval=1):
                """derivative of a numpy array"""
                diff = list()
                for i in range(len(series)):
                    value = series[i] - series[i - interval]
                    diff.append(value)
                return np.array(diff)

            def _find_turning_pts(series, z_interval, vertical=False):

                """function searches for left/right jumping points in M,
                    :param: interval is an integer representing a sequel of
                    zeros which should come to the left/right of the suspected
                    point. """

                def _argmax(lst):
                    """returns index of maximum element from tuples list by second coordinate. """
                    y_coor = [x[1] for x in lst]
                    return y_coor.index(max(y_coor))

                def _calculate_zeroes(pt, series):
                    """returns actual amount if zeroes before/after the <pt> in <series> """

                    i = 1
                    while (pt - i) >= 0 and series[pt - i] == 0:
                        i += 1
                    zeroes_left = i - 1

                    i = 1
                    while (pt + i) <= len(series) - 1 and series[pt + i] == 0:
                        i += 1
                    zeroes_right = i - 1

                    return zeroes_left, zeroes_right

                left = []
                right = []

                # checks if t is a left jumping point.
                for t in range(z_interval, len(series)):
                    if series[t] != 0:
                        for j in range(1, z_interval + 1):
                            if series[t - j] != 0:
                                break
                        if j == z_interval and series[t - z_interval] == 0:
                            left.append((t, _calculate_zeroes(t, series)[0]))

                # checks if t is a right jumping point.
                for t in range(1, len(series) - z_interval):
                    if series[t] != 0:
                        for j in range(1, z_interval + 1):
                            if series[t + j] != 0:
                                break
                        if j == z_interval and series[t + z_interval] == 0:
                            right.append((t, _calculate_zeroes(t, series)[1]))

                if vertical:
                    return left[0][0], right[-1][0]

                return left[_argmax(left)][0], right[_argmax(right)][0]

            ###

            # safety intervel to be added
            M_horizontal_safe, M_vertical_safe = 100, 130
            # number_of_zeros parameters which should come before/after a jumping point
            IZ_h, IZ_v = 50, 20

            # dictionary init
            crop_interval = {}

            # reduce img to vertical mean vector
            M_horizontal = img_diff.mean(axis=0)
            M_vertical = img_diff.mean(axis=1)

            # compute median for clipping
            horizontal_median = (M_horizontal.max() - M_horizontal.min()) / 2
            vertical_median = (M_vertical.max() - M_vertical.min()) / 2

            # clipper
            horizontal_clip = lambda t: 0 if t - horizontal_median / 2 < 0 else t
            vertical_clip = lambda t: 0 if t - vertical_median / 2 < 0 else t

            # apply clipper
            M_horizontal_mod = np.array([horizontal_clip(t) for t in M_horizontal])
            M_vertical_mod = np.array([vertical_clip(t) for t in M_vertical])

            # find suspected points for the main objects' boundaries
            right_h, left_h = _find_turning_pts(abs(_derivative(M_horizontal_mod)), IZ_h, vertical=False)
            right_v, left_v = _find_turning_pts(abs(_derivative(M_vertical_mod)), IZ_v, vertical=True)

            # bug
            right_v = 0 if right_v > len(M_vertical) / 4 else right_v

            # adding safety interval to the final result
            crop_interval[0] = (max(0, right_h - M_horizontal_safe), min(len(M_horizontal_mod), left_h + M_horizontal_safe))
            crop_interval[1] = (max(0, right_v - M_vertical_safe), min(len(M_vertical_mod), left_v + M_vertical_safe))

            return crop_interval

        cameras = list(blob2crop.keys())
        result_dict = {}

        for cam in cameras:
            num_pics = len(blob2crop[cam])
            grey = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in blob2crop[cam]]
            ind2img = dict(zip([x for x in range(num_pics)], [img for img in blob2crop[cam]]))

            for ind in ind2img.keys():
                diff_ind = list(np.argsort([abs(x - ind) for x in [x for x in range(len(grey))]])[:3])
                d_im1, d_im2 = _compute_diffs(grey[diff_ind[1]], grey[diff_ind[0]], grey[diff_ind[2]])

                hl1, hr1 = _find_crop_indices(d_im1)[0]
                hl2, hr2 = _find_crop_indices(d_im2)[0]
                vl1, vr1 = _find_crop_indices(d_im1)[1]
                vl2, vr2 = _find_crop_indices(d_im2)[1]

                hlc, hrc = min(hl1, hl2), max(hr1, hr2)
                vlc, vrc = min(vl1, vl2), max(vr1, vr2)

                result_dict[blob2crop[cam][ind]] = (hlc, hrc, vlc, vrc)

        return result_dict

    def create_mask(self, height, width, hsv_params=((0, 51, 0), (179, 255, 255)), dilate_iter=2, dilate_ker=3, erode_iter=3, erode_ker=3, apply_mask=False, save_to_local=False, rescale=False, plot=False):

        def _pad(img, h, w):
            #  in case when you have odd number
            top_pad = int(np.floor((h - img.shape[0]) / 2))
            bottom_pad = int(np.ceil((h - img.shape[0]) / 2))
            right_pad = int(np.ceil((w - img.shape[1]) / 2))
            left_pad = int(np.floor((w - img.shape[1]) / 2))
            return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant',
                                  constant_values=255))

        def _mask_function(img_path, hlc, hrc, vlc, vrc, method='hsv'):

            def _erode_function(img, mask, ker, ite):
                kernel = np.ones((ker, ker), np.uint8)
                mask_erosion = cv2.erode(mask, kernel, iterations=ite)
                imask_erosion = mask_erosion == 0
                g_img = np.zeros_like(img, np.uint8)
                g_img.fill(255)
                g_img[imask_erosion] = img[imask_erosion]

                return g_img, mask_erosion, img

            def _dilate_function(img, mask, ker, ite):
                kernel = np.ones((ker, ker), np.uint8)
                mask_dilate = cv2.dilate(mask, kernel, iterations=ite)
                imask_dilate = mask_dilate == 0
                greened_img = np.zeros_like(img, np.uint8)
                greened_img.fill(255)
                greened_img[imask_dilate] = img[imask_dilate]

                return greened_img, mask_dilate, img

            # mask by slicing the green spectrum
            image = cv2.imread(img_path)[vlc:vrc, hlc:hrc]  # cv2 read & crop

            # compute mask
            # HSV
            if method == "hsv":
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to hsv
                mask = cv2.inRange(hsv, hsv_params[0], hsv_params[1])  # mask by slicing the green spectrum

            # Grayscale
            if method == "grayscale":
                grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                mask = cv2.inRange(grayscale, 0, 230)

            # apply the mask:
            g_img = np.zeros_like(image, np.uint8)
            mask_g = np.zeros_like(image, np.uint8)
            g_img.fill(255)
            g_img[mask > 0] = image[mask > 0]
            mask_g[~mask > 0] = image[~mask > 0]
            mask_g[mask_g != 0] = 255

            # erosion and dilation
            g_img, mask, image = _dilate_function(image, mask_g, dilate_ker, dilate_iter)
            g_img, mask, image = _erode_function(image, mask, erode_ker, erode_iter)

            return g_img, mask, image

        container_list = sorted(glob.glob(self.local_dir + "/*"))

        for container_path in container_list:

            d = self.organize_container(container_path)

            N = int(math.ceil(len([item for sublist in list(d.values()) for item in sublist])) / 4)
            M = 4

            dict_w_crop = self.crop_session(d)

            if plot:
                fig, axs = plt.subplots(nrows=N, ncols=M, figsize=(20, 20))
                fig.suptitle(f'cropping results {container_path}', fontsize=20, y=1)

            for i, j in product(range(N), range(M)):

                try:
                    img_path, crop = dict_w_crop.popitem()
                    hlc, hrc, vlc, vrc = crop
                except KeyError:
                    continue

                # apply mask: Green HSV,dilate,erode
                if apply_mask:
                    img, _, _ = _mask_function(img_path, hlc, hrc, vlc, vrc)
                else:
                    img = cv2.imread(img_path)[vlc:vrc, hlc:hrc]

                # rescaling image to WxH by padding
                if rescale:
                    img = _pad(img, h=height, w=width)

                # plotting
                if plot:
                    axs[i, j].set_title(img_path.split("/")[-1])
                    axs[i, j].imshow(img)

                # save to directory
                if save_to_local:

                    if not os.path.exists(self.target_dir + "/" + img_path.split("/")[-2]): os.makedirs(
                        self.target_dir + "/" + img_path.split("/")[-2])

                    cv2.imwrite(self.target_dir + "/" + img_path.split("/")[-2] + "/" + img_path.split("/")[-1], img)
            if plot:
                plt.show()
