import cv2
import os
import numpy as np

# Paths to folders
input_folder = r'C:\Users\Laptop\Extracted_Circles_from_OMR'

def classify_circles(input_folder):
    # Output folders
    output_empty_folder = rf'{input_folder}\empty_circles'
    output_filled_folder = rf'{input_folder}\filled_circles'
    output_letter_folder = rf'{input_folder}\letter_circles'

    # Create output directories if they do not exist
    os.makedirs(output_empty_folder, exist_ok=True)
    os.makedirs(output_filled_folder, exist_ok=True)
    os.makedirs(output_letter_folder, exist_ok=True)

    # Threshold to differentiate between filled and empty circles
    threshold = 160 # Adjust this value based on your circles
    num_empty = 0
    num_filled = 0
    num_letter = 0
    # Process each extracted circle
    for circle_file in os.listdir(input_folder):
        if circle_file.endswith('.png'):
            circle_path = os.path.join(input_folder, circle_file)


            # Load the circle image
            circle_img = cv2.imread(circle_path, cv2.IMREAD_GRAYSCALE)

            # Resize for consistency if needed (optional)
            # circle_img = cv2.resize(circle_img, (50, 50))

            # Compute the average intensity
            avg_intensity = np.mean(circle_img)
            # print(f"Circle {circle_file} has an average intensity of {avg_intensity}")

            # Classify the circle
            if avg_intensity > 200:
                num_empty += 1
                # Empty circle (higher intensity)
                save_path = os.path.join(output_empty_folder, circle_file)

            elif avg_intensity > threshold:
                num_letter += 1
                # Circle with letter (medium intensity)
                save_path = os.path.join(output_letter_folder, circle_file)
                print(save_path, "with intensity", avg_intensity)

            else:
                num_filled += 1
                # Filled circle (lower intensity)
                save_path = os.path.join(output_filled_folder, circle_file)

            print(circle_file, "classified as", save_path)
            # Save the image to the corresponding folder
            cv2.imwrite(save_path, circle_img)

    print("Circles classified and saved into separate folders!")
    # print(f"Empty circles: {num_empty}")
    # print(f"Filled circles: {num_filled}, at {output_filled_folder}")
    # print(f"Letter circles: {num_letter}")

if __name__ == "__main__":
    classify_circles(input_folder)