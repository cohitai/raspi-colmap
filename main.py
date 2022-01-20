import time
import argparse
from preprocess import MaskAzure
from reconstruction import DockerizedColmap
from postprocess import FlushAzure
from supplements import PcdOps
import logging
import sys
import os

# create logger for 'raspi-colmap'
logging.getLogger('raspi-colmap')
logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.DEBUG)

# set local structure
WORKING_DIRECTORY = os.getcwd()
DATA_DIR = WORKING_DIRECTORY + "/data"
RAW_DATA_DIR = DATA_DIR + "/d1"
MASKED_DATA_DIR = DATA_DIR + "/d2"
COLMAP_OUTPUT_DIR = DATA_DIR + "/d3"
SUPP_OUTPUT_DIR = DATA_DIR + "/d4"


def main():

    parser = argparse.ArgumentParser(description="Raspi colmap 3D reconstruction.")
    parser.add_argument("-A", "--automate", help="automate server by time", nargs='+', default=10000, type=int)
    parser.add_argument("-M", "--mask", help="mask parameters", nargs='+', type=int)

    args = parser.parse_args()

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
            container_name = extractor.fetch_last_container()
            # debug_container=["1639656186-281117"]
            # apply mask and save to local (2)
            logging.info("Applying Mask on data")
            extractor.create_mask(height=720, width=1080, hsv_params=((0, 51, 0), (179, 255, 255)), dilate_iter=3,
                                  dilate_ker=3, erode_iter=3, erode_ker=6, apply_mask=True, rescale=True,
                                  save_to_local=True, plot=False)

            # reconstruct (3)
            logging.info("COLMAP reconstruction:")
            colmap_client = DockerizedColmap(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR)
            colmap_client.reconstruct()

            # upload to azure
            logging.info("Upload to Azure:")
            flusher = FlushAzure(COLMAP_OUTPUT_DIR, container_name)
            flusher.flush()

            # download dense.ply to local and apply post-operations (4)
            logging.info("Post-process with open3d")
            PcdOps(SUPP_OUTPUT_DIR, container_name).fetch_dense_from_remote()
            PcdOps(SUPP_OUTPUT_DIR, container_name).reconstruct(outliers=True, poisson=True)

            # flush new files to container
            logging.info("Uploading to Azure, poisson and inlier point cloud.")
            flusher.flush_from([SUPP_OUTPUT_DIR + "/" + "dense_inlier.ply", SUPP_OUTPUT_DIR + "/" + "poisson.ply"])

            # go to pause
            logging.info("Going to sleep..")
            time.sleep(args.automate[0])


if __name__ == '__main__':
    main()
