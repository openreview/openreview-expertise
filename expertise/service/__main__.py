import argparse
from expertise.service.server import app
import os
from google.cloud import storage

def copy_directory(bucket_name, source_blob_prefix, destination_dir):
    """Copy all files from a GCS bucket directory to a local directory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_prefix)
    for blob in blobs:
        destination_path = os.path.join(destination_dir, os.path.relpath(blob.name, start=source_blob_prefix))
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        blob.download_to_filename(destination_path)
        print(f"Copied {blob.name} to {destination_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--container', default=False, type=bool)
    args = parser.parse_args()

    if args.container:
        # Extract the bucket name and path from the environment variable
        aip_storage_uri = os.getenv('AIP_STORAGE_URI')
        if not aip_storage_uri:
            raise ValueError("AIP_STORAGE_URI environment variable is not set")

        # Assuming AIP_STORAGE_URI is in the format gs://bucket_name/path_to_directory
        bucket_name = aip_storage_uri.split('/')[2]

        # The directory to copy the artifacts to, and the subdirectory name you want
        destination_dir = "/app/expertise-utils" ## TODO: Remove this hardcode

        copy_directory(bucket_name, 'expertise-utils', destination_dir)

    app.run(host=args.host, port=args.port)