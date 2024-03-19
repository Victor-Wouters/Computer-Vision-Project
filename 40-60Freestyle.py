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
            if between(cap, 0, 11000):
                
                # Again grabbing the Fish with HSV and RGB mask
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

                # Segment size for each color change
                segment_size = (10000 - 0) // (7 * 13)  # 7 colors, repeated 13 times

                # Determine the segment
                segment = (int(cap.get(cv2.CAP_PROP_POS_MSEC)) - 0) // segment_size

                # Assign the segment to a rainbow color, cycling through the colors 13 times
                if segment % 7 == 0:
                    new_color = [0, 0, 255]  # BGR for red
                elif segment % 7 == 1:
                    new_color = [0, 127, 255]  # BGR for orange
                elif segment % 7 == 2:
                    new_color = [0, 255, 255]  # BGR for yellow
                elif segment % 7 == 3:
                    new_color = [0, 255, 0]  # BGR for green
                elif segment % 7 == 4:
                    new_color = [255, 0, 0]  # BGR for blue
                elif segment % 7 == 5:
                    new_color = [130, 0, 75]  # BGR for indigo 
                elif segment % 7 == 6:
                    new_color = [180, 0, 130]  # BGR for violet

                # Create a color layer with the same dimensions as the frame, but filled with the new color
                color_layer = np.zeros_like(frame)
                color_layer[:] = new_color

                # Use the final mask to put the color layer onto the original frame
                # Only the areas with mask=255 will be colored
                frame = np.where(final_mask[:, :, np.newaxis] == 255, color_layer, frame)
                
            if between(cap, 12000, 20000):
                
                # Same as before
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

                # Keep track of height and width of frame.
                height, width = frame.shape[:2]

                # Create an empty array of the same shape as the frame to hold the shifted pixels
                shifted_frame = np.zeros_like(frame)

                # Calculate the safe shift to avoid index out of range errors, shift 230 pixels down and 230 pixels to the left
                safe_shift_down = min(230, height)
                safe_shift_left = max(0, min(230, width - 1))

                # Copy pixels from 230 pixels lower and 230 pixels to the left to their new position
                # Avoid accessing out of bounds
                shifted_frame[:height-safe_shift_down, :width-safe_shift_left] = frame[safe_shift_down:height, safe_shift_left:width]

                # Where the mask=255 replace the original frame's pixel
                # with the corresponding pixel from 'shifted_frame'
                frame = np.where(final_mask[:, :, np.newaxis] == 255, shifted_frame, frame)
                
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
