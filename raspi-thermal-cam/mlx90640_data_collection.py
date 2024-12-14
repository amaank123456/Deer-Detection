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
from scipy import ndimage

i2c = busio.I2C(board.SCL, board.SDA, frequency=800000) # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c) # begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ # set refresh rate

# Constants
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
DURATION = 3
OUTPUT_FILE = 'no_deer_data_grayscale/no_deer_11.mp4'
t_array = []

# Initialize storage for frames and timestamps
frames = []
timestamps = []

# Capture frames
print("Start recording")
start_time = time.time()
while (time.time() - start_time) < DURATION:
    # t1 = time.monotonic()
    try:
        # Initialize the frame array
        frame = np.zeros((FRAME_HEIGHT * FRAME_WIDTH,))
        mlx.getFrame(frame) # read MLX temperatures into frame var
        data_array = (np.reshape(frame,(FRAME_HEIGHT, FRAME_WIDTH))) # reshape to 24x32
        data_array = np.fliplr(data_array)

        # Normalize temperature to 8-bit grayscale
        norm_array = cv2.normalize(data_array, None, 0, 255, cv2.NORM_MINMAX)
        norm_array = np.uint8(norm_array)
        norm_array = np.expand_dims(np.uint8(norm_array), axis=0)
        norm_array = np.expand_dims(np.uint8(norm_array), axis=0)

        # Apply colormap
        # color_frame = cv2.applyColorMap(norm_array, cv2.COLORMAP_JET)

        # Store frame and timestamps
        frames.append(norm_array)
        timestamps.append(time.time())
    except ValueError:
        continue # if error, just read again

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
    video_writer.write(frame[0][0])

# Release resources
video_writer.release()
print(f"Video written to {OUTPUT_FILE}")

s = OUTPUT_FILE.split(".")[0]+".npy"
np.save(s, np.concatenate(frames, axis=0))
print(f"Data written to {s}")
