import cv2

# Open a video file or capture device (e.g., webcam)
cap = cv2.VideoCapture('.\data\orginal_billgates.mp4')

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP42')  # You can change the codec based on your needs
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

# Loop through the frames and save the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Perform any processing on the frame if needed
    # For example, you can convert the frame to grayscale:
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, column, _ = frame.shape
    column = int(column/2)
    frame = frame[:, :column, :]
    # Write the frame to the output video file
    out.write(frame)

    # Display the frame if you want to show the video while saving
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close any open windows
cv2.destroyAllWindows()