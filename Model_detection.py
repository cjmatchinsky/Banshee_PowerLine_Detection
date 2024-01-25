import cv2
from ultralytics import YOLO

# Path to the YOLOv8 model weights
#
path_to_model = 'dataset/runs/detect/Yolov8n_powerline_model_weights_test_e80/weights/best.pt'

# Load the YOLOv8 model
#
model = YOLO(path_to_model, 'v8')

# Open the video capture object (using camera index 1)
#
cap = cv2.VideoCapture(1)

# Loop through the video frames
#
while cap.isOpened():
    
    # Read a frame from the video
    #
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        #
        results = model(frame)

        # Visualize the results on the frame
        #
        annotated_frame = results[0].plot()

        # Display the annotated frame
        #
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        #
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
