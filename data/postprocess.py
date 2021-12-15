import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, AccessPolicy, ContainerSasPermissions, PublicAccess
from glob import glob

os.environ['AZURE_STORAGE_CONNECTION_STRING_2'] = 'DefaultEndpointsProtocol=https;AccountName=pointclouds1;AccountKey=2+G3qwZZXh13ShcSpoUHxFxj6i/3YYTurFibVeKoVBI6HrwdOHwc2sgEQydY5VTzSwavVRiFrL5Uf5MQSF4oFA==;EndpointSuffix=core.windows.net'
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING_2')
# Create the BlobServiceClient object which will be used to create a container client.
blob_service_client_2 = BlobServiceClient.from_connection_string(connect_str)


class AzureFlush:
    def __init__(self, pcd_dir, container_name):
        self.pcd_dir = pcd_dir
        self.container_name = container_name

    def flush(self):
        # Create the container.
        # container_client = blob_service_client_2.create_container(self.container_name)
        files = glob(self.pcd_dir + '/' + '*.ply')


#a = AzureFlush("/home/liteandfog/raspi-colmap/data/d3", "").flush()
#print(a)