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
    # Get the frame rate of the video
    #fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    # Calculate the duration in seconds
    #duration_seconds = total_frames / fps

    #print(f"The video is {duration_seconds} seconds long.")
    # while loop where the real work happens
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
                lower_color_bound = np.array([10, 60, 50])
                upper_color_bound = np.array([30, 255, 255])

                # Create a mask with the specified color range
                mask_hsv = cv2.inRange(hsv_image, lower_color_bound, upper_color_bound)

                kernel = np.ones((5,5),np.uint8)

                # Improve HSV mask
                mask_hsv_improved = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
                mask_hsv_improved = cv2.morphologyEx(mask_hsv_improved, cv2.MORPH_OPEN, kernel)

                # Step 4: Apply the mask
                frame = cv2.bitwise_and(frame, frame, mask=mask_hsv_improved)
                
                RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                lower_brown_rgb = np.array([200, 0, 0])
                upper_brown_rgb = np.array([255, 0, 0])

                # Create mask directly in RGB space
                mask_rgb = cv2.inRange(RGB_image, lower_brown_rgb, upper_brown_rgb)
                
                kernel = np.ones((5,5),np.uint8)
                mask_rgb_improved = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)
                mask_rgb_improved = cv2.morphologyEx(mask_rgb_improved, cv2.MORPH_OPEN, kernel)


                final_mask = cv2.bitwise_or(mask_hsv_improved,mask_rgb_improved)
                # Step 4: Apply the mask
                frame = cv2.bitwise_and(frame, frame, mask=final_mask)
                
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
