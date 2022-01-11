import time

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
    # preprocess (1)
    extractor = MaskAzure(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR, SUPP_OUTPUT_DIR)
    # clean the working directory
    extractor.init()
    # dl last container to local
    container_name = extractor.fetch_last_container()

    # container_name = ""

    # apply mask and save to local (2)
    extractor.create_mask(height=720, width=1080, hsv_params=((0, 51, 0), (179, 255, 255)), dilate_iter=2, erode_iter=3, apply_mask=True, rescale=True, save_to_local=True, plot=False)

    # reconstruct (3)
    colmap_client = DockerizedColmap(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR)
    colmap_client.reconstruct()

    # upload to azure
    flusher = FlushAzure(COLMAP_OUTPUT_DIR, container_name)
    flusher.flush()

    # download dense.ply to local and apply post-operations (4)
    PcdOps(SUPP_OUTPUT_DIR, container_name).fetch_dense_from_remote()
    PcdOps(SUPP_OUTPUT_DIR, container_name).reconstruct(outliers=True, poisson=True)

    # flush new files to container
    flusher.flush_from([SUPP_OUTPUT_DIR + "/" + "dense_inlier.ply", SUPP_OUTPUT_DIR + "/" + "poisson.ply"])


if __name__ == '__main__':
    while True:
        main()
        logging.info("going to sleep..")
        time.sleep(10000)
