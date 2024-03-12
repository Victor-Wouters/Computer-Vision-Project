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
            if between(cap, 0, 5000):
                
                # do something using OpenCV functions (skipped here so we simply write the input frame back to output)
                # Step 2: Convert to HSV
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Horizontal edges
                sobel_horizontal = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

                # Vertical edges
                sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                # Convert edge maps to positive values
                abs_sobel_horizontal = cv2.convertScaleAbs(sobel_horizontal)
                abs_sobel_vertical = cv2.convertScaleAbs(sobel_vertical)
    
                
                final_mask = cv2.bitwise_or(abs_sobel_horizontal,abs_sobel_vertical)
                _, edges_thresholded = cv2.threshold(final_mask, 100, 255, cv2.THRESH_BINARY)
                #edges_colored = cv2.cvtColor(edges_thresholded, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR
                #frame = cv2.addWeighted(frame, 0.9, edges_thresholded, 0.1, 0)
                #frame[edges_thresholded == 255] = [0, 0, 255]

                red_edges = np.zeros_like(frame)
                red_edges[edges_thresholded == 255] = [0, 0, 255]
                frame = red_edges
                #frame = cv2.bitwise_and(frame, frame, mask=edges_thresholded)
            if between(cap, 5001, 10000):
                
                # do something using OpenCV functions (skipped here so we simply write the input frame back to output)
                # Step 2: Convert to HSV
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Horizontal edges
                sobel_horizontal = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

                # Vertical edges
                sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                # Convert edge maps to positive values
                abs_sobel_horizontal = cv2.convertScaleAbs(sobel_horizontal)
                abs_sobel_vertical = cv2.convertScaleAbs(sobel_vertical)
    
                
                final_mask = cv2.bitwise_or(abs_sobel_horizontal,abs_sobel_vertical)
                _, edges_thresholded = cv2.threshold(final_mask, 100, 255, cv2.THRESH_BINARY)
                #edges_colored = cv2.cvtColor(edges_thresholded, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR
                #frame = cv2.addWeighted(frame, 0.9, edges_thresholded, 0.1, 0)
                #frame[edges_thresholded == 255] = [0, 0, 255]

                red_edges = np.zeros_like(frame)
                red_edges[edges_thresholded == 255] = [0, 0, 255]
                frame = red_edges
                #frame = cv2.bitwise_and(frame, frame, mask=edges_thresholded)
                
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

#python Bear.py -i bear1.mp4 -o bear2.mp4