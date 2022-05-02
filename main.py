import time
import argparse
from preprocess import MaskAzure
from reconstruction import DockerizedColmap
from postprocess import FlushAzure
from supplements import PcdOps
import logging
import sys
import os


# set local structure
WORKING_DIRECTORY = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIRECTORY, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "d1")  # raw images input
MASKED_DATA_DIR = os.path.join(DATA_DIR, "d2")  # images dir (after masking)
COLMAP_OUTPUT_DIR = os.path.join(DATA_DIR, "d3")  # colmap output dir
SPARSE_OUTPUT_DIR = os.path.join(COLMAP_OUTPUT_DIR, "sparse/0")   # colmap sparse output dir
SUPP_OUTPUT_DIR = os.path.join(DATA_DIR, "d4")  # Azure containers files
DB_PATH = os.path.join(COLMAP_OUTPUT_DIR, "database.db")


# create logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)],
                    level=logging.DEBUG)

logging.getLogger('raspi-config')


def main():

    parser = argparse.ArgumentParser(description="Raspi colmap 3D reconstruction.")
    parser.add_argument("-A", "--automate", help="automate server by time", default=10000, type=int)
    parser.add_argument("-D", "--debug", help="debug mode", default=False, type=bool)
    args = parser.parse_args()

    debug_kwargs = {"container_name": None,
                    "extract_images": False,
                    "compute_mask": False,
                    "compute_reconstruction": False,
                    "upload_to_azure": False,
                    "compute_postprocessor": False
                    }

    if args.automate:
        while True:

            # preprocess (1)
            logging.info("STARTING AUTOMATION:")
            extractor = MaskAzure(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR, SUPP_OUTPUT_DIR)
            # clean the working directory
            logging.info("Clean the workspace.")
            extractor.init()

            # dl last container to local
            logging.info("Extracting data from Azure.")
            # container_name = ["1651357151-954354"]
            container_name = extractor.fetch_last_container()

            # apply mask and save to local (2)
            logging.info("Applying Mask on data")
            mask_kwargs = {"height": 720,
                           "width": 1080,
                           "hsv_params": ((0, 50, 0), (179, 255, 255)),
                           "dilate_iter": 0,
                           "dilate_ker": 2,
                           "erode_iter": 10,
                           "erode_ker": 4,
                           "apply_mask": True,
                           "rescale": True,
                           "save_to_local": True,
                           "plot": False
                           }
            extractor.create_mask(**mask_kwargs)

            # reconstruct (3)
            logging.info("COLMAP reconstruction:")
            colmap_client = DockerizedColmap(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR, "colmap/colmap:latest")
            colmap_client.reconstruct()

            # upload to azure
            logging.info("Upload to Azure:")
            flusher = FlushAzure(COLMAP_OUTPUT_DIR, SPARSE_OUTPUT_DIR, DB_PATH, container_name)
            flusher.flush()

            # download dense.ply to local and apply post-operations (4)
            logging.info("Post-process with open3d")
            PcdOps(SUPP_OUTPUT_DIR, container_name).fetch_file_from_remote('dense.ply')
            PcdOps(SUPP_OUTPUT_DIR, container_name).reconstruct(outliers=True, poisson=True)

            # upload files to Azure container.
            logging.info("Uploading to Azure, poisson and inlier point cloud.")
            flusher.flush_from([os.path.join(SUPP_OUTPUT_DIR, "dense_inlier.ply"),
                                os.path.join(SUPP_OUTPUT_DIR, "poisson.ply")])

            # fetch sparse_dir files
            PcdOps(SUPP_OUTPUT_DIR, container_name).fetch_file_from_remote('cameras.txt')
            PcdOps(SUPP_OUTPUT_DIR, container_name).fetch_file_from_remote('images.txt')
            PcdOps(SUPP_OUTPUT_DIR, container_name).fetch_file_from_remote('points3D.txt')
            PcdOps(SUPP_OUTPUT_DIR, container_name).fetch_file_from_remote('project.ini')

            # go to pause
            logging.info("Going to sleep..")
            time.sleep(args.automate)


if __name__ == '__main__':
    main()
