from ultralytics import YOLO
import os

# Set the YOLO model
model = YOLO('yolov8n')

# Change to the YOLO directory
yolo_dir = "dataset"
os.chdir(yolo_dir)

# Continuous training loop
# while True:
#     try:
# Train the YOLO model
results = model.train(
    data='data.yaml',
    imgsz=640,
    epochs=210,
    batch=16,
    name='Yolov8n_powerline_model_weights_test_e210',
)

# Print a message after successful training
print('Training completed successfully!')

# Uncomment the following line if you want to break out of the loop on success
# break

# except Exception as e:
#     print(f"Error: {e}")
#     print("Restarting training...")
