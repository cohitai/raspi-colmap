from preprocess import MaskAzure
from reconstruction import DockerizedColmap
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


def main():
    # preprocess
    extractor = MaskAzure(RAW_DATA_DIR, MASKED_DATA_DIR)
    # clean the working directory
    extractor.init()
    # dl last container to local
    extractor.fetch_last_container()
    # apply mask and save to local
    extractor.create_mask(height=720, width=1080, apply_mask=True, rescale=True, save_to_local=True, plot=False)

    # reconstruct
    #colmap_client = DockerizedColmap(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR)
    #colmap_client.reconstruct(outliers=True, poisson=True)

    # upload to azure


if __name__ == '__main__':
    main()
