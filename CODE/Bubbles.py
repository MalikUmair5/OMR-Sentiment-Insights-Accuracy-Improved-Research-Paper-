import cv2
import numpy as np
import os

# Load the image
image_folder = r'C:\Users\Laptop\OMR'
output_folder = r'C:\Users\Laptop\Extracted_Circles_from_OMR'
#detected_circles_folder = r'C:\Users\Laptop\Detected_Circles'

def extract_circles(image_folder, output_folder):
    # Create output directories if they do not exist
    os.makedirs(output_folder, exist_ok=True)
    #os.makedirs(detected_circles_folder, exist_ok=True)

    num= 1
    for image_path in os.listdir(image_folder):
        if image_path.endswith('.png') or image_path.endswith('.jpg'):
            print(num, os.path.join(image_folder, image_path))
            num += 1 
            # Read the image
            image = cv2.imread(os.path.join(image_folder, image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detect circles using Hough Circle Transform
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,  # Inverse ratio of resolution
                minDist=100,  # Minimum distance between detected centers
                param1=500,  # Higher threshold for Canny edge detection
                param2=28,  # Accumulator threshold for circle centers
                minRadius=18,  # Minimum radius of circles
                maxRadius=23  # Maximum radius of circles
            )

            # Initialize counter for saving images
            circle_counter = 0

            # If circles are detected
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Extract circle coordinates and radius
                    x, y, radius = i[0], i[1], i[2]

                    # Define the bounding box for the circle
                    x_start, x_end = max(0, x - radius), min(image.shape[1], x + radius)
                    y_start, y_end = max(0, y - radius), min(image.shape[0], y + radius)

                    # Crop the circle from the image
                    circle_roi = image[y_start:y_end, x_start:x_end]

                    # Save the filled circle as an individual image
                    circle_counter += 1
                    circle_path = os.path.join(output_folder, f'circle{num}_{circle_counter}.png')
                    cv2.imwrite(circle_path, circle_roi)
                    print(f"Saved circle {circle_counter} to {circle_path}")

                    # Optionally, draw circles for debugging (uncomment to visualize)
                    cv2.circle(image, (x, y), radius, (0, 255, 0), 2)

            # Save the image with detected circles
            #cv2.imwrite(os.path.join(detected_circles_folder, f'detected_circles_{image_path}'), image)

            # Save and display the result
            print(f"Extracted {circle_counter} filled circles and saved them to {output_folder}.")
        # Uncomment the lines below to visualize the debug image locally
        # cv2.imshow('Detected Circles', image)             # Show image
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_circles(image_folder, output_folder)