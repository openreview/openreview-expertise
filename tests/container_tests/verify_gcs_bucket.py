# Save this as test_gcs_emulator.py
import os
import sys
import json
from google.cloud import storage

def test_gcs_connection():
    print("Testing GCS emulator connection")
    # Print environment variables
    print(f"STORAGE_EMULATOR_HOST: {os.environ.get('STORAGE_EMULATOR_HOST')}")
    print(f"GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    
    try:
        # Initialize client
        storage_client = storage.Client()
        print("Created storage client")
        
        # List buckets
        print("Listing buckets:")
        buckets = list(storage_client.list_buckets())
        print(f"Found {len(buckets)} buckets:")
        for bucket in buckets:
            print(f"- {bucket.name}")
        
        # Make sure our test bucket exists
        bucket_name = "test-bucket"
        print(f"Checking if bucket '{bucket_name}' exists...")
        try:
            bucket = storage_client.get_bucket(bucket_name)
            print(f"Bucket {bucket_name} exists!")
            
            # Try to upload a test file
            print("Uploading test file...")
            blob = bucket.blob("test_file.txt")
            blob.upload_from_string("This is a test file")
            print("Test file uploaded successfully!")
            
            # Verify the file was uploaded
            print("Verifying file exists in bucket...")
            blob_list = list(bucket.list_blobs())
            print(f"Files in bucket: {[b.name for b in blob_list]}")
            
            return True
            
        except Exception as e:
            print(f"Error accessing bucket: {e}")
            return False
            
    except Exception as e:
        print(f"Error connecting to GCS emulator: {e}")
        return False

if __name__ == "__main__":
    success = test_gcs_connection()
    sys.exit(0 if success else 1)