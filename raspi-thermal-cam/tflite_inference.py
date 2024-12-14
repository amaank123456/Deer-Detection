##########################################
# MLX90640 Thermal Camera w Raspberry Pi
# -- 2Hz Sampling with Simple Routine
##########################################
#
import time,board,busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
from collections import deque
import RPi.GPIO as GPIO

i2c = busio.I2C(board.SCL, board.SDA, frequency=800000) # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c) # begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ # set refresh rate

# Setup the LED
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(21,GPIO.OUT)

# Constants
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
DURATION = 120
t_array = []
OUTPUT_FILE = "test_inference_gray.mp4"

# Initialize storage for frames, timestamps, and inference queue
frames = []
timestamps = []
inference_queue = deque()

# Load the TensorFlow Lite model
model_path = "model/quantized_tflite_model_grayscale_v2.tflite"  # Change to your model path
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Capture frames
print("Start recording")
start_time = time.time()
while (time.time() - start_time) < DURATION:
    try:
        # Initialize the frame array
        frame = np.zeros((FRAME_HEIGHT * FRAME_WIDTH,))
        mlx.getFrame(frame) # read MLX temperatures into frame var
        data_array = (np.reshape(frame,(FRAME_HEIGHT, FRAME_WIDTH))) # reshape to 24x32
        data_array = np.fliplr(data_array)

        # Normalize temperature to 8-bit grayscale
        norm_array = cv2.normalize(data_array, None, 0, 255, cv2.NORM_MINMAX)
        norm_array = np.uint8(norm_array)

        # Expand image array frames
        color_frame = np.expand_dims(norm_array, axis=0)
        color_frame = np.expand_dims(color_frame, axis=0)
        color_frame = color_frame.transpose(0,2,3,1)
        
        # Perform inference
        interpreter.set_tensor(input_index, color_frame)
        interpreter.invoke()

        # Retrieve output data
        output = interpreter.tensor(output_index)
        out = np.argmax(output()[0])

        # Store the inference in a queue and print the output
        inference_queue.append(out)
        if len(inference_queue) == 8:
            if sum(inference_queue) >= 4:
                GPIO.output(21,GPIO.HIGH)
            else:
                GPIO.output(21,GPIO.LOW)
            inference_queue.popleft()
        print(out)

        # Store frame and timestamps
        frames.append(color_frame[0])
        timestamps.append(time.time())

        # break
    except ValueError:
        continue # if error, just read again

# Shut off power
GPIO.output(21,GPIO.LOW)

# Calculate FPS from timestamps
if len(timestamps) > 1:
    elapsed_times = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    avg_frame_time = sum(elapsed_times) / len(elapsed_times)
    fps = 1 / avg_frame_time
else:
    fps = 1

print("End recording")
print(f"Calculated FPS: {fps}")

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT), isColor=False)

for frame in frames:
    video_writer.write(frame)

# Release resources
video_writer.release()
print(f"Video written to {OUTPUT_FILE}")

