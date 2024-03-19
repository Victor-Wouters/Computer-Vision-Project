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
    
   
    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            if between(cap, 1000, 9000):
                      
                hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # HSV image
            
                lower_color_bound = np.array([25, 100, 100])
                upper_color_bound = np.array([35, 255, 255])

                mask_hsv = cv2.inRange(hsv_image, lower_color_bound, upper_color_bound)
                mask_hsv_original = mask_hsv.copy()

                kernel = np.ones((5,5),np.uint8)

                mask_hsv_improved = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)     # HSV improvements
                mask_hsv_improved = cv2.morphologyEx(mask_hsv_improved, cv2.MORPH_OPEN, kernel)
                
                RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # RGB image

                lower_brown_rgb = np.array([150, 150, 0])
                upper_brown_rgb = np.array([255, 255, 100])

                mask_rgb = cv2.inRange(RGB_image, lower_brown_rgb, upper_brown_rgb)
                mask_rgb_original = mask_rgb.copy()
                
                kernel = np.ones((5,5),np.uint8)
                mask_rgb_improved = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)     # RGB improvements
                mask_rgb_improved = cv2.morphologyEx(mask_rgb_improved, cv2.MORPH_OPEN, kernel)

                mask_hsv_combined = np.where(mask_hsv_improved > mask_hsv_original, 2, mask_hsv_original)   # Keep track of improvements
                mask_rgb_combined = np.where(mask_rgb_improved > mask_rgb_original, 2, mask_rgb_original)

                basic_mask_combined = cv2.bitwise_or(mask_hsv_original, mask_rgb_original)      # combine HSV and RGB
                improved_mask_combined = cv2.bitwise_or(mask_hsv_combined, mask_rgb_combined)
                
                color_white = [255, 255, 255]
                color = [0, 255, 0]

                final_image = np.zeros_like(frame)                                              # Visualize with black background, white mask and green improvements
                final_image[basic_mask_combined > 0] = color_white
                improvement_areas = cv2.subtract(improved_mask_combined, basic_mask_combined)
                final_image[improvement_areas > 0] = color
             
                frame = final_image
                
                
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
