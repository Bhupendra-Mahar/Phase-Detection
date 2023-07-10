import cv2
import os

def extract_frames(video_path,output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    frame_interval = 100

    while True:
        # Read the next frame
        ret, frame = video.read()

        # If frame reading was not successful, break the loop
        if not ret:
            break
        output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_path, frame)
        # Display the frame when the frame count reaches the interval
        if frame_count % frame_interval == 0:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        frame_count += 1

    # Release the video object and close any open windows
    video.release()
    cv2.destroyAllWindows()

# Provide the path to your video file
video_path = "Z:\\New folder (6)\\cholec80\\videos\\video01.mp4"
output_dir = "Z:\\New folder (6)\\Output_images"
# Call the function to extract and view frames with a gap of 100 frames
extract_frames(video_path,output_dir)
