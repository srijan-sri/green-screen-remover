# Import necessary libraries
import cv2  # OpenCV library for image and video processing
import numpy as np  # NumPy library for numerical operations
import tkinter as tk  # Tkinter library for creating GUI applications
from tkinter import filedialog  # Module for file dialog in Tkinter
from PIL import Image, ImageTk  # PIL for image processing and displaying images in Tkinter

# Define the range of green color in HSV (Hue, Saturation, Value)
lower_green = np.array([35, 50, 50])  # Lower bound of green in HSV
upper_green = np.array([85, 255, 255])  # Upper bound of green in HSV

# Initialize video capture and background image variables
cap = None  # Variable to hold the video capture object
background_image = None  # Variable to hold the loaded background image

# Function to update the HSV range based on user input from sliders
def update_hsv():
    global lower_green, upper_green  # Use global variables for HSV bounds
    lower_green = np.array([hue_min.get(), sat_min.get(), val_min.get()])  # Update lower bound
    upper_green = np.array([hue_max.get(), sat_max.get(), val_max.get()])  # Update upper bound

# Function to load a background image from the file system
def load_background():
    global background_image  # Use global variable for the background image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])  # Open file dialog
    if file_path:  # Check if a file was selected
        background_image = cv2.imread(file_path)  # Read the image using OpenCV
        background_image = cv2.resize(background_image, (640, 480))  # Resize the image to match video frame size

# Function to start video capture from the webcam
def start_capture():
    global cap  # Use global variable for the video capture object
    if cap is None:  # Check if video capture is not already running
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize video capture from the default webcam
        update_video()  # Start updating video frames

# Function to stop video capture
def stop_capture():
    global cap  # Use global variable for the video capture object
    if cap is not None:  # Check if video capture is running
        cap.release()  # Release the video capture object
        cap = None  # Reset the variable

# Function to update the video frame and apply green screen removal
def update_video():
    if cap is not None:  # Check if video capture is running
        ret, frame = cap.read()  # Read a frame from the webcam
        if ret:  # Check if the frame was successfully captured
            frame = cv2.resize(frame, (640, 480))  # Resize the frame to a fixed size
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert the frame to HSV color space
            mask = cv2.inRange(hsv_frame, lower_green, upper_green)  # Create a mask for the green color
            kernel = np.ones((5, 5), np.uint8)  # Define a kernel for morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
            mask = cv2.bilateralFilter(mask, 9, 75, 75)  # Smooth the mask using a bilateral filter
            inverse_mask = cv2.bitwise_not(mask)  # Invert the mask to get non-green areas
            foreground = cv2.bitwise_and(frame, frame, mask=inverse_mask)  # Extract non-green areas from the frame
            if background_image is not None:  # Check if a background image is loaded
                background_extract = cv2.bitwise_and(background_image, background_image, mask=mask)  # Extract green areas from the background
            else:
                background_extract = np.zeros_like(frame)  # Use a black background if no image is loaded
            final_image = cv2.add(foreground, background_extract)  # Combine the foreground and background
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)  # Convert the final image to RGB for Tkinter
            img = Image.fromarray(final_image_rgb)  # Convert the image to a PIL Image
            imgtk = ImageTk.PhotoImage(image=img)  # Convert the PIL Image to a Tkinter-compatible image
            video_label.imgtk = imgtk  # Store the image reference to prevent garbage collection
            video_label.configure(image=imgtk)  # Update the label with the new image
    root.after(10, update_video)  # Schedule the function to run again after 10ms

# Create the main GUI window
root = tk.Tk()  # Initialize the Tkinter root window
root.title("Green Screen Removal")  # Set the title of the window

# Create a label to display the video
video_label = tk.Label(root)  # Create a label widget for displaying video frames
video_label.pack()  # Add the label to the window

# Create sliders for HSV range adjustment
hue_min = tk.Scale(root, label="Hue Min", from_=0, to=179, orient=tk.HORIZONTAL)  # Slider for minimum hue
hue_min.set(35)  # Set default value
hue_min.pack()  # Add the slider to the window

hue_max = tk.Scale(root, label="Hue Max", from_=0, to=179, orient=tk.HORIZONTAL)  # Slider for maximum hue
hue_max.set(85)  # Set default value
hue_max.pack()  # Add the slider to the window

sat_min = tk.Scale(root, label="Saturation Min", from_=0, to=255, orient=tk.HORIZONTAL)  # Slider for minimum saturation
sat_min.set(50)  # Set default value
sat_min.pack()  # Add the slider to the window

sat_max = tk.Scale(root, label="Saturation Max", from_=0, to=255, orient=tk.HORIZONTAL)  # Slider for maximum saturation
sat_max.set(255)  # Set default value
sat_max.pack()  # Add the slider to the window

val_min = tk.Scale(root, label="Value Min", from_=0, to=255, orient=tk.HORIZONTAL)  # Slider for minimum value
val_min.set(50)  # Set default value
val_min.pack()  # Add the slider to the window

val_max = tk.Scale(root, label="Value Max", from_=0, to=255, orient=tk.HORIZONTAL)  # Slider for maximum value
val_max.set(255)  # Set default value
val_max.pack()  # Add the slider to the window

# Create a button to update the HSV range
update_button = tk.Button(root, text="Update HSV Range", command=update_hsv)  # Button to update HSV range
update_button.pack()  # Add the button to the window

# Create a button to load a background image
load_button = tk.Button(root, text="Load Background Image", command=load_background)  # Button to load background image
load_button.pack()  # Add the button to the window

# Create a button to start video capture
start_button = tk.Button(root, text="Start Capture", command=start_capture)  # Button to start video capture
start_button.pack()  # Add the button to the window

# Create a button to stop video capture
stop_button = tk.Button(root, text="Stop Capture", command=stop_capture)  # Button to stop video capture
stop_button.pack()  # Add the button to the window

# Start the GUI event loop
root.mainloop()  # Start the Tkinter main event loop

# Release the video capture and close OpenCV windows when the GUI is closed
if cap is not None:  # Check if video capture is running
    cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows