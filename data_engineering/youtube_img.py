from pytube import YouTube
import cv2
import os


# Function to download YouTube video
def download_video(url, path):
    yt = YouTube(url)
    # Get video title and sanitize it for use as a filename
    video_title = yt.title.replace(' ', '_').replace('/', '_')
    # Select the highest resolution stream available
    ys = yt.streams.get_highest_resolution()
    # Download the video and save it with the video title as filename
    video_path = os.path.join(path, f"{video_title}.mp4")
    ys.download(output_path=path, filename=f"{video_title}.mp4")
    return video_path

# Function to extract frames from the video
def extract_frames(video_path, frames_dir, every_n_frames=30):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    video_title = os.path.basename(video_path).replace('.mp4', '')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frames every 'every_n_frames' frames
        if frame_idx % every_n_frames == 0:
            frame_path = os.path.join(frames_dir, f"{video_title}_frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved {frame_path}")
        frame_idx += 1
    cap.release()
    print("Done extracting frames.")

def process_videos(file_path, base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    video_path = os.path.join(base_path, 'videos')
    frames_path = os.path.join(base_path, 'frames')

    if not os.path.exists(video_path):
        os.makedirs(video_path)
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    # Read URLs from file
    with open(file_path, 'r') as file:
        urls = file.readlines()

    for url in urls:
        url = url.strip()
        if url:
            print(f"Processing video: {url}")
            video_filepath = download_video(url, video_path)
            extract_frames(video_filepath, frames_path)

# Paths configuration
file_path = 'videos.txt'
base_path = r'/image_data/'

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"Error: '{file_path}' does not exist.")
else:
    process_videos(file_path, base_path)
