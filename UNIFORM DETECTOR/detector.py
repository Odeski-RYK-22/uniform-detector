import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uniform Detector")

        # Set the size of the main window
        self.root.geometry("1200x700")

        # Create and pack the Start/Stop button
        self.start_button = ttk.Button(root, text="Start Video", command=self.toggle_video)
        self.start_button.pack(pady=10)

        # Create a frame for the video and status display
        self.display_frame = ttk.Frame(root)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        # Create and pack the video frame (initially empty)
        self.video_frame = ttk.Label(self.display_frame)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create and pack the text box for displaying the uniform status
        self.text_box = tk.Text(self.display_frame, height=35, width=40)
        self.text_box.pack(side=tk.RIGHT, padx=10, pady=10)

        # Initialize variables
        self.video_running = False
        self.cap = None

        # Load the pre-trained object detection model
        self.net = cv2.dnn.readNetFromCaffe(
            'MobileNetSSD_deploy.prototxt.txt',
            'MobileNetSSD_deploy.caffemodel'
        )

        # Define the list of class labels MobileNet SSD was trained to detect
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                        "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                        "tvmonitor"]

        # Load uniform images for boy and girl
        self.boy_uniform_img = Image.open("uniform/boy.jpg")
        self.girl_uniform_img = Image.open("uniform/girl.jpg")

        # Resize uniform images to fit display
        self.boy_uniform_img = self.boy_uniform_img.resize((200, 300))
        self.girl_uniform_img = self.girl_uniform_img.resize((200, 300))

    def toggle_video(self):
        if self.video_running:
            self.stop_video()
        else:
            self.start_video()

    def start_video(self):
        # Replace with your IP webcam's URL
        self.cap = cv2.VideoCapture("http://192.168.170.44:8080/video")
        self.video_running = True
        self.start_button.config(text="Stop Video")
        self.update_frame()

    def stop_video(self):
        self.video_running = False
        self.start_button.config(text="Start Video")
        if self.cap:
            self.cap.release()
        self.video_frame.config(image="")
        self.text_box.delete(1.0, tk.END)

    def update_frame(self):
        if self.video_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize the frame to fit the window size
                frame = cv2.resize(frame, (800, 600))

                # Detect objects in the frame
                status, detected_items = self.detect_objects(frame)

                # Draw green box around detected person
                for item in detected_items:
                    if item['label'] == 'person':
                        x, y, w, h = item['bbox']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the uniform status only if a person is detected
                if any(item['label'] == 'person' for item in detected_items):
                    self.display_status(status, detected_items)
                else:
                    self.text_box.delete(1.0, tk.END)
                    self.video_frame.config(image="")

                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.config(image=imgtk)

            self.root.after(10, self.update_frame)

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        detected_items = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    detected_items.append({'label': 'person', 'bbox': (startX, startY, endX - startX, endY - startY)})

        # Dummy detected items for demonstration purposes
        detected_items.extend([
            {'label': 'T-shirt', 'bbox': (100, 100, 150, 150)},
            {'label': 'pants', 'bbox': (100, 250, 150, 300)},
            {'label': 'shoes', 'bbox': (100, 450, 150, 500)}
        ])

        status, message = self.check_colors(frame, detected_items)
        return status, message

    def is_color_in_range(self, hsv_image, lower_range, upper_range):
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        return cv2.countNonZero(mask) > 0

    def check_colors(self, frame, detected_items):
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_ranges = {
            'white': (np.array([0, 0, 200]), np.array([180, 30, 255])),
            'black': (np.array([0, 0, 0]), np.array([180, 255, 50]))
        }

        status = []
        for item in detected_items:
            x, y, w, h = item['bbox']
            roi = hsv_image[y:y+h, x:x+w]
            if item['label'] == 'T-shirt' and not self.is_color_in_range(roi, *color_ranges['white']):
                status.append('Not Allowed: T-shirt color is incorrect')
            if item['label'] == 'pants' and not self.is_color_in_range(roi, *color_ranges['black']):
                status.append('Not Allowed: Pants color is incorrect')
            if item['label'] == 'shoes' and not self.is_color_in_range(roi, *color_ranges['black']):
                status.append('Not Allowed: Shoes color is incorrect')

        if not status:
            status.append('Uniform is correct')
        return status, detected_items

    def display_status(self, status, detected_items):
        self.text_box.delete(1.0, tk.END)

        # Determine if boy or girl uniform based on detected items
        has_tshirt = any(item['label'] == 'T-shirt' for item in detected_items)
        if has_tshirt:
            uniform_img = self.boy_uniform_img  # Assume boy's uniform if T-shirt detected
            self.text_box.insert(tk.END, "Boy's Uniform Detected\n\n")
        else:
            uniform_img = self.girl_uniform_img  # Assume girl's uniform if no T-shirt detected
            self.text_box.insert(tk.END, "Girl's Uniform Detected\n\n")

        uniform_img_tk = ImageTk.PhotoImage(uniform_img)
        self.video_frame.imgtk = uniform_img_tk
        self.video_frame.config(image=uniform_img_tk)

        self.text_box.insert(tk.END, "\n".join(status) + "\n\nDetected Items:\n")
        for item in detected_items:
            self.text_box.insert(tk.END, f"{item['label']}: {item['bbox']}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
