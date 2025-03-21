import io
from azure.storage.blob import BlobServiceClient
import csv
import json
import pandas as pd
import os
import numpy as np

from utils.classes.Settings import Settings

def load_csv_dict(filepath):
    """Load CSV data from a file and return a list of rows."""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        return [row for row in reader]

def load_csv_list(filepath):
    """Load CSV data from a file and return a list of rows."""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        return [row for row in reader]
    
def load_csv_df(filepath, header=None):
    return pd.read_csv(filepath, delimiter=';', header=header)
    
def load_excel_list(filepath):
    """Load Excel data from a file and return a list of rows as dictionaries."""
    df = pd.read_excel(filepath)
    return df.to_dict(orient='records')

def load_excel_from_azure(file_name):
    file_stream = load_file_stream_from_azure(file_name)
    df = pd.read_csv(io.BytesIO(file_stream))
    return df

def load_dict_from_azure(file_name):
    file_stream = load_file_stream_from_azure(file_name)

    if file_stream is None:
        return None

    dictionary = json.load(io.BytesIO(file_stream))
    return dictionary

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

def save_dictionary_to_azure(file_name: str, dictionary: dict):
    blob_service_client = BlobServiceClient.from_connection_string(Settings().AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(Settings().CONTAINER_NAME)
    blob_client = container_client.get_blob_client(file_name)

    json_data = json.dumps(dictionary)
    blob_client.upload_blob(json_data, overwrite=True)

def load_csv_np(filepath, skip_header):
    return np.genfromtxt(filepath, delimiter=";", skip_header=skip_header)

def save_csv(filename, rows):
    """Save a list of rows to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(rows)