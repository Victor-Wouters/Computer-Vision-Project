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
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            if between(cap, 0, 10000):
                
                # do something using OpenCV functions (skipped here so we simply write the input frame back to output)
                # Step 2: Convert to HSV
                
                hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Step 3: Define the range of your target color in HSV
                lower_color_bound = np.array([25, 100, 100])
                upper_color_bound = np.array([35, 255, 255])

                # Create a mask with the specified color range
                mask_hsv = cv2.inRange(hsv_image, lower_color_bound, upper_color_bound)

                kernel = np.ones((5,5),np.uint8)

                # Improve HSV mask
                mask_hsv_improved = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
                mask_hsv_improved = cv2.morphologyEx(mask_hsv_improved, cv2.MORPH_OPEN, kernel)

                # Step 4: Apply the mask
                #frame = cv2.bitwise_and(frame, frame, mask=mask_hsv_improved)
                
                RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                lower_brown_rgb = np.array([150, 150, 0])
                upper_brown_rgb = np.array([255, 255, 100])

                # Create mask directly in RGB space
                mask_rgb = cv2.inRange(RGB_image, lower_brown_rgb, upper_brown_rgb)
                
                kernel = np.ones((5,5),np.uint8)
                mask_rgb_improved = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)
                mask_rgb_improved = cv2.morphologyEx(mask_rgb_improved, cv2.MORPH_OPEN, kernel)


                final_mask = cv2.bitwise_or(mask_hsv_improved,mask_rgb_improved)
                # Step 4: Apply the mask
                # Calculate the segment size for each color change
                segment_size = (10000 - 0) // (7 * 13)  # 7 colors, repeated 4 times

                # Determine the segment based on the current 'cap' value
                segment = (int(cap.get(cv2.CAP_PROP_POS_MSEC)) - 0) // segment_size

                # Map the segment to a rainbow color, cycling through the colors 4 times
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
                    new_color = [130, 0, 75]  # BGR for indigo (approximation)
                elif segment % 7 == 6:
                    new_color = [180, 0, 130]  # BGR for violet (approximation)

                # Create a color layer with the same dimensions as the frame, but filled with the new color
                color_layer = np.zeros_like(frame)
                color_layer[:] = new_color

                # Use the final mask to blend the color layer onto the original frame
                # Only the areas with mask=255 will be colored
                frame = np.where(final_mask[:, :, np.newaxis] == 255, color_layer, frame)
                #frame = cv2.bitwise_and(frame, frame, mask=final_mask)
                
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
