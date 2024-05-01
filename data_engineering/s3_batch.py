import boto3
import os

client = boto3.client('s3',
                      aws_access_key_id = "",
                      aws_secret_access_key = '')

path = 'image_data\\frames'
for file in os.listdir(path):
    upload_file_bucket = 'aws-data-warehouse'
    upload_file_location = 'data/youtube_videos/' + str(file)
    client.upload_file(os.path.join(path, file), upload_file_bucket, upload_file_location)