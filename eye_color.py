import cv2
import numpy as np

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Define a dictionary of eye color ranges
color_ranges = {
    'brown': ([0, 70, 90], [10, 255, 255]),
    'blue': ([90, 70, 90], [130, 255, 255]),
    'green': ([30, 70, 90], [70, 255, 255])
}

# Open a live video feed from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) which contains the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop through the detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Extract the region of interest (ROI) which contains the eye
            eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Convert the eye ROI to the HSV color space
            hsv = cv2.cvtColor(eye_roi_color, cv2.COLOR_BGR2HSV)

            # Compute the mean color of the eye ROI in the HSV color space
            mean_color = cv2.mean(hsv)

            # Convert the mean color to the RGB color space
            rgb_color = cv2.cvtColor(np.uint8([[mean_color[:3]]]), cv2.COLOR_HSV2BGR)[0][0]

            # Determine the dominant eye color based on the RGB color
            dominant_color = None
            for color, (lower, upper) in color_ranges.items():
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                if cv2.inRange(np.array([rgb_color]), lower, upper).all():
                    dominant_color = color
                    break

            # Draw a rectangle around the eye and display the dominant eye color
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(frame, dominant_color or 'unknown', (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the detected eyes and the dominant eye color
    cv2.imshow('Live Eye Color Detector', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()