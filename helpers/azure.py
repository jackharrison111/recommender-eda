

def download_from_fileshare():
    ...

def upload_to_fileshare():
    ...

def download_from_blob():

     # Instantiate a BlobServiceClient using a connection string
   from azure.storage.blob import BlobServiceClient
   blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

   # Instantiate a ContainerClient
   container_client = blob_service_client.get_container_client("mynewcontainer")

    ...

def upload_to_blob():
    ...

def download_from_cosmos():
    ...

def upload_to_cosmos():
    ...


