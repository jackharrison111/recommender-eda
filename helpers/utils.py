import os
from typing import List

from azure.storage.blob import BlobServiceClient


class DownloadFromWeights:

    def __init__(self, blob_container_name):
        self.blob_conn_string = "INSERT"
        # self.blob_conn_string = os.environ["BLOB_WEIGHTS_CONN_STR"]
        # self.blob_container_name = self.config.get("blob").get("weights_container")
        self.blob_container_name = blob_container_name

        # Create the BlobServiceClient object which will be used to create a container client
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.blob_conn_string
        )

    def download_images(self, images_filenames: List[str], images_filepaths: List[str]):
        
        file_names = images_filenames
        paths = images_filepaths
        for file_name, path in zip(file_names, paths):
            # Create a local directory to hold blob data
            # os.mkdir(path)
            # base_file_name = file_name.rsplit("/", 1)[1]
            blob_client = self.blob_service_client.get_blob_client(
                container=self.blob_container_name, blob=file_name
            )
            with open(os.path.join(path, file_name), "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

    def list_blobs(self):

        blob_container = self.blob_service_client.get_container_client(self.blob_container_name)
        blobs_list = blob_container.list_blobs()
        return blobs_list
        

if __name__ == "__main__":
    obj = DownloadFromWeights("instagram")

    obj.download_images(["instagram.com_collector_20211030_151127.errors.csv"] , ["./"])