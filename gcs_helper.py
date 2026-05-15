import os
from google.cloud import storage


def get_bucket():
    """Connects to GCP and retrieves your specific bucket."""
    client = storage.Client()
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    return client.bucket(bucket_name)


def upload_to_gcs(local_path: str, gcs_path: str):
    """Uploads a file from your local /tmp folder to the Cloud Bucket."""
    bucket = get_bucket()
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to GCS as {gcs_path}")


def download_from_gcs(gcs_path: str, local_path: str):
    """Downloads a file from the Cloud Bucket to your local /tmp folder."""
    bucket = get_bucket()
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        return False

    # Make sure the local directory exists before downloading
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_path} to {local_path}")
    return True