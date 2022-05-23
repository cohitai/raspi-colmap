import time
import argparse
from preprocess import MaskAzure
from reconstruction import DockerizedColmap
from poses import Poses
from postprocess import FlushAzure
from supplements import PcdOps
import logging
import sys
import os


# set local structure
WORKING_DIRECTORY = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIRECTORY, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # raw images input
COLMAP_OUTPUT_DIR = os.path.join(DATA_DIR, "output")  # colmap output dir
MASKED_DATA_DIR = os.path.join(COLMAP_OUTPUT_DIR, "images")  # images dir (after masking)
SPARSE_OUTPUT_DIR = os.path.join(COLMAP_OUTPUT_DIR, "sparse/0")   # colmap sparse output dir
SUPP_OUTPUT_DIR = os.path.join(DATA_DIR, "supplements")  # Azure containers files
DB_PATH = os.path.join(COLMAP_OUTPUT_DIR, "database.db")


# create logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)],
                    level=logging.DEBUG)

logging.getLogger('raspi-config')


def main():

    parser = argparse.ArgumentParser(description="Raspi colmap 3D reconstruction.")
    parser.add_argument("-A", "--automate", help="automate server by time", default=False, type=bool)
    parser.add_argument("-C", "--costume", help="costume run", default='', type=str)
    parser.add_argument("-D", "--debug", help="debug mode", default=False, type=bool)
    parser.add_argument("-T", "--sleeptime", help="servers sleeping time", default=10000, type=int)
    parser.add_argument("--black_bkg", help="automate server by time", action='store_true')
    parser.add_argument("--clean_workspace", help="delete all data directories in the working space",
                        action='store_true')
    args = parser.parse_args()

    debug_kwargs = {"container_name": None,
                    "extract_images": False,
                    "compute_mask": False,
                    "compute_reconstruction": False,
                    "upload_to_azure": False,
                    "compute_postprocessor": False
                    }

    dirs_kargs = {
                  "raw_dir": RAW_DATA_DIR,
                  "output_dir": COLMAP_OUTPUT_DIR,
                  "images_dir": MASKED_DATA_DIR,
                  "supp_dir": SUPP_OUTPUT_DIR
                  }

    mask_kwargs = {"height": 720,
                   "width": 1080,
                   "hsv_params": ((0, 6, 0), (80, 255, 255)),
                   "dilate_iter": 1,
                   "dilate_ker": 2,
                   "erode_iter": 5,
                   "erode_ker": 4,
                   "apply_mask": True,
                   "rescale": True,
                   "save_to_local": True,
                   "plot": False,
                   "black_background": args.black_bkg
                   }

    if args.clean_workspace:
        extractor = MaskAzure(**dirs_kargs)
        # clean the working directory
        logging.info("Clean the workspace.")
        extractor.init()

    if args.automate:
        while True:

            # preprocess (1)
            logging.info("STARTING AUTOMATION:")
            extractor = MaskAzure(**dirs_kargs)
            # clean the working directory
            logging.info("Clean the workspace.")
            extractor.init()

            # dl last container to local
            logging.info("Extracting data from Azure.")
            # container_name = ["1651357151-954354"]
            container_name = extractor.fetch_last_container()

            # apply mask and save to local (2)
            logging.info("Applying Mask on data")
            extractor.create_mask(**mask_kwargs)

            # reconstruct (3)

            colmap_kargs = {"feature_extractor": True,
                            "exhaustive_matcher": True,
                            "mapper": True,
                            "image_undistorter": True,
                            "patch_match_stereo": True,
                            "stereo_fusion": True,
                            "model_converter": True}

            logging.info("COLMAP reconstruction:")
            colmap_client = DockerizedColmap(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR, "colmap/colmap:latest")
            colmap_client.reconstruct(**colmap_kargs)

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
            time.sleep(args.sleeptime)

    if args.costume:
        logging.info(f"Begin the Customized Run.")
        logging.info(f"Container Name: {args.costume}")

        # preprocess (1)
        extractor = MaskAzure(**dirs_kargs)

        # 1.1 clean the working directory
        logging.info("Clean the workspace.")
        extractor.init()

        # 1.2 dl last container to local
        logging.info("Extracting data from Azure.")
        extractor.fetch_last_container([args.costume])

        # apply mask and save to local (2)
        logging.info("Applying Mask on data")
        extractor.create_mask(**mask_kwargs)

        # reconstruct (3)
        # colmap session configuration
        colmap_kargs = {"feature_extractor": True,
                    "exhaustive_matcher": True,
                    "mapper": True,
                    "image_undistorter": False,
                    "patch_match_stereo": False,
                    "stereo_fusion": False,
                    "model_converter": True}
        logging.info("COLMAP reconstruction:")
        colmap_client = DockerizedColmap(RAW_DATA_DIR, MASKED_DATA_DIR, COLMAP_OUTPUT_DIR, "colmap/colmap:latest")
        colmap_client.reconstruct(**colmap_kargs)

        # poses extraction (4)
        Poses(COLMAP_OUTPUT_DIR, MASKED_DATA_DIR).run()


if __name__ == '__main__':
    main()
