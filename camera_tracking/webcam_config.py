import cv2
import argparse
'''_________________________________________________________________________________________________________________________________________________________________________'''
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args
'''_________________________________________________________________________________________________________________________________________________________________________'''
args = parse_arguments()
frame_width, frame_height = args.webcam_resolution

'''_________________________________________________________________________________________________________________________________________________________________________'''
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_height)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

def list_webcams():
    # Iterate over all possible webcam indices
    for i in range(10):  # You can adjust the range based on your system
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam {i}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            cap.release()





