import tflite_runtime.interpreter as tflite
import numpy as np
import cv2  # Optional, for image handling
import time
import matplotlib.pyplot as plt

# Load the TensorFlow Lite model
model_path = "model/quantized_tflite_model_grayscale_v2.tflite"  # Change to your model path
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Prepare input data (example: image preprocessing)
def preprocess_image(image_path):
    # Read and resize image to match model input size
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # input_data = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(image, axis=0).astype(np.uint8)  # Add grayscale dimension
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    input_data = input_data.transpose(0,2,3,1)
    return input_data

# Path to the input image
image_path = "model/0018.png"  # Replace with your image path
input_data = preprocess_image(image_path)

# Perform inference
start_time = time.time()
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()

# Retrieve output data
# output = interpreter.get_tensor(output_index)
output = interpreter.get_tensor(output_index)
out = np.argmax(output)

# Print the output
print(f"Inference time: {(time.time() - start_time)*1000} ms")
print("Model output:", out)
