import cv2
import numpy as np
import math
import time
import os

def detect_skin(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the hand)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(max_contour)
            return True, (x, y, w, h), mask

    return False, None, mask

# Initialize webcam
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
counter = 0

# Set the folder for saving images
folder = "Data/Okay"

# Make sure the folder exists
os.makedirs(folder, exist_ok=True)

print("Starting data collection for sign language...")
print("Press 's' to save an image, 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to get frame from webcam")
        break

    imgOutput = img.copy()

    # Detect hand using skin color
    hand_detected, bbox, skin_mask = detect_skin(img)

    if hand_detected:
        x, y, w, h = bbox

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Make sure the crop region is within the image bounds
        y_start = max(0, y-offset)
        y_end = min(img.shape[0], y + h + offset)
        x_start = max(0, x-offset)
        x_end = min(img.shape[1], x + w + offset)

        imgCrop = img[y_start:y_end, x_start:x_end]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Draw rectangle and label
        cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imgOutput, "Hand Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the cropped and processed images
        cv2.imshow('Hand Crop', imgCrop)
        cv2.imshow('Processed Hand', imgWhite)

    # Show the skin detection mask
    cv2.imshow('Skin Mask', skin_mask)

    # Show the main output
    cv2.imshow('Hand Detection', imgOutput)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Data collection completed")