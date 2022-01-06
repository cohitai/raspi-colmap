import os
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.storage.blob import BlobServiceClient, AccessPolicy, ContainerSasPermissions, PublicAccess
from glob import glob
import logging
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# Azure Storage Account: pointclouds1
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING_2')
# Create the BlobServiceClient object which will be used to create a container client.
blob_service_client_2 = BlobServiceClient.from_connection_string(connect_str)


class FlushAzure:
    def __init__(self, pcd_dir, container_name):
        self.pcd_dir = pcd_dir
        self.container_name = container_name

    def flush(self):

        # Relevant files
        files = glob(self.pcd_dir + '/' + '*.ply')

        # Create container
        try:
            container_client = blob_service_client_2.create_container(self.container_name)
        except ResourceExistsError:
            logging.info(f"Container {self.container_name} already exists.")
            timestamp = str(datetime.timestamp(datetime.now())).replace(".", "-")
            self.container_name = self.container_name + "-" + timestamp
            logging.info(f"Container new name: {self.container_name}.")
            container_client = blob_service_client_2.create_container(self.container_name)


        # Create access policy
        access_policy = AccessPolicy(permission=ContainerSasPermissions(read=True, write=True))
        identifiers = {'read': access_policy}

        # Specifies full public read access for container and blob data.
        public_access = PublicAccess.Container

        # Set the access policy on the container
        container_client.set_container_access_policy(signed_identifiers=identifiers, public_access=public_access)

        # Upload files to blob storage
        for file in files:

            # fetch file name
            _, file_name = os.path.split(file)

            # Create a blob client using the local file name as the name for the blob.
            blob_client = blob_service_client_2.get_blob_client(container=self.container_name, blob=file_name)
            logging.info(f"\nUploading to Azure Storage as blob:\n\t{file_name}")

            # Upload the created img.
            with open(file, "rb") as data:
                blob_client.upload_blob(data)
