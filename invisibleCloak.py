# Learning about the image processing techniques, such as saturation/
# We are going to write a program to make an invisibilty cloak
# Here were are using camera vission library (cv2)
# The program will make the the things invisible based on the color of the object/user

# Importing all modules
import cv2
import numpy as np

# Starting the webcam to read/capture the bg
video = cv2.VideoCapture(0)
image = cv2.imread("me.jpeg") 

# While this is true
while True:
    # Return the frame after being read
    ret, frame = video.read()
    # Print
    print(frame)
    # Resize the frame and image
    frame = cv2.resize(frame,(640,480))
    image = cv2.resize(image, (640,480))

    # Upper and Lower saturation of Black
    l_black = np.array([30,30, 0])
    u_black = np.array([104,153, 70])

    # Masking the black parts of the image
    # Getting the results with the black part
    mask = cv2.inRange(frame, l_black, u_black)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    # Total frame
    f = frame - res
    f = np.where(f == 0, image, f)

    # Show the users the output of the frame without the black parts
    # Show the users the mask
    cv2.imshow("Video", frame)
    cv2.imshow("Mask", f)

    # If the wait is 1 second and the dimensions are these, then break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and destroy all the windows
video.release()
cv2.destroyAllWindows()