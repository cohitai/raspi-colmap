from preprocess import MaskAzure
from reconstruction import DockerizedColmap
from postprocess import FlushAzure

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


# frame's size:
HT, WH = 720, 1080

# set PATHS for colmap (to be added later to .env)
LOCAL_SERVER_DIR = '/home/liteandfog/raspi-colmap'
SERVER_RAW_DIR = LOCAL_SERVER_DIR + '/data/d1'
SERVER_MASKED_DIR = LOCAL_SERVER_DIR + '/data/d2'
SERVER_COLMAP_OUTPUT_DIR = LOCAL_SERVER_DIR + '/data/d3'


# BUG: stucks at poisson - cant open PLY file
# for the Dockerized application: 2 containers app:
#                1. principal container raspi/colmap with these files.
#                2. colmap:test container
#                shared volumes:
#                docker daemon local to docker daemon in container (1).
#                /data/ local to /data/ in container (1).
#                /data/ local to /data/ in container (2).
# running with: docker run -v /var/run/docker.sock:/var/run/docker.sock  -v /home/liteandfog/raspi-colmap/data:/app/data --runtime=nvidia raspi/reconstructor:latest



def main():

    # preprocess
    extractor = MaskAzure(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR)
    # clean the working directory
    extractor.init()
    # dl last container to local
    container_name = extractor.fetch_last_container()
    # apply mask and save to local
    extractor.create_mask(height=HT, width=WH, apply_mask=True, rescale=True, save_to_local=True, plot=False)

    # reconstruct
    colmap_client = DockerizedColmap(SERVER_RAW_DIR, SERVER_MASKED_DIR, SERVER_COLMAP_OUTPUT_DIR)
    colmap_client.reconstruct(outliers=True, poisson=True)

    # upload to azure
    FlushAzure(COLMAP_OUTPUT_DIR, container_name).flush()


if __name__ == '__main__':
    main()
