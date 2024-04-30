import boto3
import os
client = boto3.client('s3',
                      aws_access_key_id = 
                      aws_secret_accesss_key = )

for file in os.listdr():
    upload_file_bucket = 'aws-data-warehouse'
    upload_file_location = 'data/youtube_videos/' + str(file)
    client.upload_file(file, upload_file_bucket, upload_file_location)
