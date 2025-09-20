import io
from azure.storage.blob import BlobServiceClient
import csv
import json
import pandas as pd
import os
import numpy as np

from utils.classes.Settings import Settings

def load_df_from_azure(file_name):
    file_stream = load_file_stream_from_azure(file_name)
    df = pd.read_csv(io.BytesIO(file_stream))
    return df

def load_file_stream_from_azure(file_name):
    blob_service_client = BlobServiceClient.from_connection_string(Settings().AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(Settings().CONTAINER_NAME)
    blob_client = container_client.get_blob_client(file_name)

    try:
        stream = blob_client.download_blob()
    except:
        return None

    file_stream = stream.readall()

    return file_stream

def save_csv_to_azure(file_name: str, df: pd.DataFrame):
    blob_service_client = BlobServiceClient.from_connection_string(Settings().AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(Settings().CONTAINER_NAME)
    blob_client = container_client.get_blob_client(file_name)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)