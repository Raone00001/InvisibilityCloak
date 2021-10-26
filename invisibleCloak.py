# Learning about the image processing techniques, such as saturation/
# We are going to write a program to make an invisibilty cloak
# Here were are using camera vission library (cv2)
# The program will make the the things invisible based on the color of the object/user

# Importing all modules
import cv2
import numpy as py
import time

# Storing/Saving the background into a file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

# Starting the webcam to read/capture the bg
cap = cv2.VideoCapture(0)

# Warming up the system
time.sleep(2)
bg = 0

# Capture the bg for 60 frames (for a clearer way)
for i in range(60):
    ret,bg = cap.read()
    
# Flipping the bg that we captured
bg = np.flip(bg,axis=1)

# Reading the captured frame until the camera is open
while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    img = np.flip(img,axis=1)

    # As we are capturing the frames, we are capturing the colors within the frames. 
    # So we are going to convert BGR(blue, green, red) to HSV(hue, saturation, value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generating a mask to detect the red color
    # Intensity of red
    lower_red = np.array([0,120,50])
    upper_red = np.array([10,255,255])
    # Mask 1     
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    # Mask 2     
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Adding the values and storing it in mask 1
    mask1 = mask1 + mask2

    # Adding the diluting affect to the image
    # For that, we will be using morphologyEx(src, dst, op, kernel) 
    mask_1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Selecting the part that does not have mask 1
    mask_2 = cv2.bitwise_not(mask1)

    # Keeping only the part of images without red color
    res1 = cv2.bitwise_and(img, img, mask=mask_2)

    # Keeping only the part of images red color
    res2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    # Generating the final output by merging the 2 results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    output_file.write(final_output)

    # Displaying the output to the users
    cv2.inShow("Magic", final_output)
    cv2.waitKey(1)

# Releasing
cap.release()
out.release()
cv2.destroyAllWindows()