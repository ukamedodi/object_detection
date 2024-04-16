# Importing required libraries
import cv2  # For video and image operations
import cvzone  # For additional computer vision operations
import math  # For mathematical operations
from ultralytics import YOLO  # Importing YOLO for object detection

# Capturing video from a video file
cap = cv2.VideoCapture('roadsigns.mp4')

# Loading a pre-trained YOLO model from a saved file
model = YOLO('best.pt')

# Defining class names that correspond to classes detected by the model
classnames = ['danger', 'mandatory', 'other', 'prohibitory']

# Loop to process each frame of the video
while True:
    # Reading a frame from the video
    ret, frame = cap.read()

    # Resizing the frame for consistent processing
    frame = cv2.resize(frame, (640, 480))

    # Making a copy of the frame for operations that may modify the image
    feed = frame.copy()

    # Running the YOLO model on the frame
    results = model(frame)

    # Iterating through detection results
    for info in results:
        # Extracting bounding boxes from the results
        parameters = info.boxes
        for box in parameters:
            # Extracting coordinates and dimensions from the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1  # Width and height of the bounding box

            # Extracting the part of the frame corresponding to the bounding box
            plate_area = feed[y1:y1 + h, x1:x1 + w]

            # Retrieving the confidence score of the detection
            confidence = box.conf[0]

            # Determining the class of the detected object
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]

            # Calculating a percentage confidence
            conf = math.ceil(confidence * 100)

            # If needed, add a condition to filter detections by confidence and class
            # Drawing a rectangle around the detected object in green
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Adding text to the frame that includes the class of the object
            cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)

    # Displaying the processed frame
    cv2.imshow('frame', frame)

    # Wait for a millisecond and exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the video capture object and closing all OpenCV windows
cap.release()
cv2.destroyAllWindows()
