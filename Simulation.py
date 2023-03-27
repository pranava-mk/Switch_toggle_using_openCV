import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon, QColor, QPainter, QBrush
from PyQt5.QtCore import Qt

# Create the GUI window
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 200, 200)
        self.setWindowTitle("Object Detection")
        self.show()

    # Change the background color of the window based on the detected flag
    def set_flag(self, flag):
        self.flag = flag
        self.update()

    # Draw a circle in the window
    def paintEvent(self, event):
        painter = QPainter(self)
        brush = QBrush(QColor(Qt.green) if self.flag else QColor(Qt.red))
        painter.setBrush(brush)
        painter.drawEllipse(0, 0, self.width(), self.height())

# Load the object detection model and set the confidence threshold
model = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt" , "MobileNetSSD_deploy.caffemodel")
conf_threshold = 0.5

# Set the classes we want to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "dog", "horse", "motorbike", "person"]

# The index of the class label 'person' in the above list
PERSON_CLASS_IDX = 15

# Open the webcam
cap = cv2.VideoCapture(0)

# Create the GUI window
app = QApplication(sys.argv)
window = Window()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Create a blob from the frame and set it as input to the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    model.setInput(blob)

    # Run forward pass to get output from the model
    detections = model.forward()

    # Loop through the detections and draw a rectangle around the human objects
    flag = False
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
            flag = True

    # Update the GUI window with the flag value
    window.set_flag(flag)
    app.processEvents()


    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
window.close()
app.exit()
