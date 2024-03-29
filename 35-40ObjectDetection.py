import argparse
import cv2
import sys
import numpy as np


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            if between(cap, 0, 3000):
                
                # Same procedure used as before to grab the Fish with HSV and RGB mask
                hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                lower_color_bound = np.array([25, 100, 100])
                upper_color_bound = np.array([35, 255, 255])

                mask_hsv = cv2.inRange(hsv_image, lower_color_bound, upper_color_bound)

                kernel = np.ones((5,5),np.uint8)

                mask_hsv_improved = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
                mask_hsv_improved = cv2.morphologyEx(mask_hsv_improved, cv2.MORPH_OPEN, kernel)
                
                RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                lower_brown_rgb = np.array([150, 150, 0])
                upper_brown_rgb = np.array([255, 255, 100])

                mask_rgb = cv2.inRange(RGB_image, lower_brown_rgb, upper_brown_rgb)
                
                kernel = np.ones((5,5),np.uint8)
                mask_rgb_improved = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)
                mask_rgb_improved = cv2.morphologyEx(mask_rgb_improved, cv2.MORPH_OPEN, kernel)


                final_mask = cv2.bitwise_or(mask_hsv_improved,mask_rgb_improved)
               
                # Finding the max contours 
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
            
                # Draw a rectangle around the fish
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)

                # SAVE THE OBJECT HSV COLOR HISTOGRAM OF THE MASK OF THE LAST FRAME IN THIS IF-STATEMENT
                object_histogram = compute_histogram(hsv_image, mask=final_mask)


            if between(cap, 3001, 6000):

                hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
               
                # Initialize an empty grayscale image
                grayscale_image = np.zeros_like(frame[:, :, 0])
                
                # The size of the region that will move over the entire frame
                region_size = 5
                
                # List to store dissimilarity values for normalization
                dissimilarity_values = []

                # Loop over the frame by regions
                for y in range(0, hsv_image.shape[0], region_size):
                    for x in range(0, hsv_image.shape[1], region_size):
                        # Extract the region
                        region = hsv_image[y:y+region_size, x:x+region_size]
                        
                        # Compute the HSV color histogram for this region
                        region_histogram = compute_histogram(region)

                        # Calculate histogram similarity with compareHist
                        similarity = histogram_similarity(object_histogram, region_histogram)

                        # Keep track of the dissimilarity (inverse to similarity) 
                        dissimilarity_values.append(1 - similarity) 


                # Normalize dissimilarity values to get it in the 0-255 range
                dissimilarity_values = np.array(dissimilarity_values)
                min_dissimilarity, max_dissimilarity = dissimilarity_values.min(), dissimilarity_values.max()
                dissimilarity_values_normalized = (dissimilarity_values - min_dissimilarity) / (max_dissimilarity - min_dissimilarity) * 255

                # Update the grayscale image with normalized intensity values, update happens in same order as values stored in list
                index = 0
                for y in range(0, hsv_image.shape[0], region_size):
                    for x in range(0, hsv_image.shape[1], region_size):
                        intensity = 255 - dissimilarity_values_normalized[index]  # Inverting intensity: lower dissimilarity = higher intensity
                        grayscale_image[y:y+region_size, x:x+region_size] = intensity
                        index += 1

                # Convert the grayscale image to BGR for display
                frame = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

                
            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def compute_histogram(image, mask=None):
    
    #Compute the color histogram in the HSV space for the given image.
    histogram = cv2.calcHist([image], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return histogram

def histogram_similarity(hist1, hist2):
    
    #Calculate similarity between two histograms using compareHist
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)

# white-black scale of the similarity (compareHist) between: the HSV color histogram of the fish by using 1 frame with the fish grabbed at 2s <--> the HSV color
#histogram of every 5x5 pixels region of a frame , for all frames between during 3s