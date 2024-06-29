import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uniform Detector")
        self.root.geometry("1500x800")  # Adjusted window size for better spacing

        # Create a canvas for the background image
        self.canvas = tk.Canvas(root, width=1500, height=800)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Load and display the background image
        self.background_image = self.load_image("uniform/gate.png")
        if self.background_image is not None:
            self.background_image = Image.fromarray(self.background_image)
            self.background_photo = ImageTk.PhotoImage(self.background_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)

        # Create and place the Start/Stop button with adjustments
        self.start_button = tk.Button(
            root,
            text="Start Scanning",
            command=self.toggle_video,
            bg="blue",
            fg="white",
            font=("Helvetica", 16, "bold"),
            width=20,
            height=2,
            padx=10,
            pady=5
        )
        self.start_button.place(x=650, y=50)

        # Create a frame for the video display
        self.video_frame = tk.Label(root, bg='gray')
        self.video_frame.place(x=100, y=150, width=800, height=600)

        # Create a frame for the uniform status display
        self.status_frame = tk.Frame(root, bg='gray')
        self.status_frame.place(x=950, y=150, width=400, height=600)

        # Create the text box for displaying the uniform status
        self.text_box = tk.Text(self.status_frame, height=35, width=40, bg='white')
        self.text_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Initialize variables
        self.video_running = False
        self.cap = None

        # Load uniform images for boy and girl
        self.boy_uniform_img = self.load_image("uniform/boy.jpg")
        self.girl_uniform_img = self.load_image("uniform/girl.jpg")
        self.tshirt_img = self.load_image("uniform/tshirt.jpg")
        self.pants_img = self.load_image("uniform/pants.jpg")
        self.shoes_img = self.load_image("uniform/shoes.jpg")

        # Keep a reference to avoid garbage collection
        self.img_refs = {}

    def load_image(self, filename):
        if os.path.exists(filename):
            img = cv2.imread(filename)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Error: File {filename} not found or could not be opened.")
        return None

    def toggle_video(self):
        if self.video_running:
            self.stop_video()
        else:
            self.start_video()

    def start_video(self):
        # Replace with your IP webcam's URL
        self.cap = cv2.VideoCapture("http://192.168.170.44:8080/video")
        self.video_running = True
        self.start_button.config(text="Stop Scanning")
        self.update_frame()

    def stop_video(self):
        self.video_running = False
        self.start_button.config(text="Start Scanning")
        if self.cap:
            self.cap.release()
        self.video_frame.config(image="")
        self.text_box.delete(1.0, tk.END)

    def update_frame(self):
        if self.video_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize the frame to fit the video frame size
                frame = cv2.resize(frame, (800, 600))

                # Detect objects in the frame
                status, detected_items = self.detect_objects(frame)

                # Draw boxes around detected items
                for item in detected_items:
                    x, y, w, h = item['bbox']
                    if item['correct']:
                        color = (255, 255, 255)  # White for complete uniform
                    elif any(label in ['T-shirt', 'pants', 'shoes'] for label in item['label']):
                        color = (0, 255, 255)    # Yellow for incomplete uniform
                    else:
                        color = (0, 0, 255)      # Red for not wearing uniform
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Display the uniform status
                self.display_status(status, detected_items)

                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.config(image=imgtk)

                # Keep a reference to avoid garbage collection
                self.img_refs["frame"] = imgtk

            self.root.after(10, self.update_frame)

    def detect_objects(self, frame):
        detected_items = []
        templates = {
            'T-shirt': self.tshirt_img,
            'pants': self.pants_img,
            'shoes': self.shoes_img
        }

        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        for label, template in templates.items():
            if template is not None:
                template_resized = cv2.resize(template, (frame_resized.shape[1] // 4, frame_resized.shape[0] // 4))
                res = cv2.matchTemplate(frame_resized, template_resized, cv2.TM_CCOEFF_NORMED)
                threshold = 0.6
                loc = np.where(res >= threshold)
                if loc[0].size > 0:
                    for pt in zip(*loc[::-1]):
                        detected_items.append({'label': label, 'bbox': (pt[0]*2, pt[1]*2, template_resized.shape[1]*2, template_resized.shape[0]*2), 'correct': True})
                else:
                    detected_items.append({'label': label, 'bbox': (0, 0, 0, 0), 'correct': False})

        status, message = self.check_colors(frame, detected_items)
        return status, detected_items

    def is_color_in_range(self, hsv_image, lower_range, upper_range):
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        return cv2.countNonZero(mask) > 0

    def check_colors(self, frame, detected_items):
        required_items = ['T-shirt', 'pants', 'shoes']
        detected_labels = [item['label'] for item in detected_items if item['correct']]

        status = []

        if all(item in detected_labels for item in required_items):
            status.append('Uniform is complete')
        elif any(item in detected_labels for item in required_items):
            status.append('Incomplete Uniform: Some items are missing')
        else:
            status.append('Not Wearing Uniform: No items detected')

        return status, detected_items

    def display_status(self, status, detected_items):
        self.text_box.delete(1.0, tk.END)

        # Determine if boy or girl uniform based on detected items
        has_tshirt = any(item['label'] == 'T-shirt' for item in detected_items)
        if has_tshirt:
            uniform_img = self.boy_uniform_img
            self.text_box.insert(tk.END, "Boy's Uniform Detected\n\n")
        else:
            uniform_img = self.girl_uniform_img
            self.text_box.insert(tk.END, "Girl's Uniform Detected\n\n")

        if uniform_img is not None:
            uniform_img = Image.fromarray(uniform_img)
            uniform_img_tk = ImageTk.PhotoImage(uniform_img)
            self.text_box.imgtk = uniform_img_tk
            self.video_frame.config(image=uniform_img_tk)

            # Keep a reference to avoid garbage collection
            self.img_refs["uniform"] = uniform_img_tk

        self.text_box.insert(tk.END, "\n".join(status) + "\n\nDetected Items:\n")
        for item in detected_items:
            self.text_box.insert(tk.END, f"{item['label']}: {item['bbox']} - {'Correct' if item['correct'] else 'Incorrect'}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
