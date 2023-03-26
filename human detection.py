# includes confidence and uses MobileNetSSD

import cv2
import numpy as np

# Load the object detection model and set the confidence threshold
model = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt" , "MobileNetSSD_deploy.caffemodel")
conf_threshold = 0.5

# Set the classes we want to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person"]
# The index of the class label 'person' in the above list
PERSON_CLASS_IDX = 15

# Open the webcam
cap = cv2.VideoCapture(0)

# Set default flag value
flag = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Create a blob from the frame and set it as input to the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    model.setInput(blob)

    # Run forward pass to get output from the model
    detections = model.forward()

    # Loop through the detections and draw a rectangle around the human objects
    for i in range(detections.shape[2]):
        class_id = int(detections[0, 0, i, 1])
        confidence = detections[0, 0, i, 2]

        if class_id == PERSON_CLASS_IDX and confidence > conf_threshold:
            # Get the coordinates of the human object
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the rectangle around the human object and display the confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = "{:.2f}%".format(confidence * 100)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Set the flag if human objects are detected
            flag = 1

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('7'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Use the flag variable to trigger the flashlight of a smartphone
# if flag == 1:
    # code to turn on flashlight of smartphone
