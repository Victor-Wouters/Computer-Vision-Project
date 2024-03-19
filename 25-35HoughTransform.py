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
            if between(cap, 6000, 9000):
                
                # Parameters not strict as last
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=145, param2=22, minRadius=50, maxRadius=500)

                if circles is not None:
                    circles = np.round(circles[0,:]).astype("int")
                    for (x,y,r) in circles:
                        cv2.circle(frame, (x,y), r, (0,0,255), 4)
                        cv2.circle(frame, (x,y), 2, (0,0,255), -1)
            if between(cap, 3000, 6000):
                
                # Parameters less strict
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=165, param2=25, minRadius=50, maxRadius=500)

                if circles is not None:
                    circles = np.round(circles[0,:]).astype("int")
                    for (x,y,r) in circles:
                        cv2.circle(frame, (x,y), r, (0,0,255), 4)
                        cv2.circle(frame, (x,y), 2, (0,0,255), -1)

            if between(cap, 0, 3000):
                
                # Parameters strict as first
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=190, param2=28, minRadius=50, maxRadius=500)

                if circles is not None:
                    circles = np.round(circles[0,:]).astype("int")
                    for (x,y,r) in circles:
                        cv2.circle(frame, (x,y), r, (0,0,255), 4)
                        cv2.circle(frame, (x,y), 2, (0,0,255), -1)

                
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
