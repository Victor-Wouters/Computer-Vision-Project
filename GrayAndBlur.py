import argparse
import cv2
import sys


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
            if between(cap, 0, 1000):
                # do something using OpenCV functions (skipped here so we simply write the input frame back to output)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            if between(cap, 1000, 2000):
                pass

            if between(cap, 2000, 3000):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if between(cap, 4000, 6000):
                frame = cv2.GaussianBlur(frame, (9, 9), 0)
            
            if between(cap, 6000, 8000):
                frame = cv2.GaussianBlur(frame, (21, 21), 0)

            if between(cap, 8000, 10000):
                frame = cv2.bilateralFilter(frame, 9, 75, 75)

            if between(cap, 10000, 12000):
                frame = cv2.bilateralFilter(frame, 9, 115, 115)

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

    # python Bear.py -i bear1.mp4 -o bear2.mp4