"""
GUI Application for Handwritten Character Recognition
Allows users to draw characters and get predictions from the trained model
"""

import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, messagebox
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class HandwritingGUI:
    def __init__(self, model_path='models/best_model.h5', input_size=(28, 28)):
        """
        Initialize the GUI application
        
        Args:
            model_path: Path to trained model
            input_size: Size of input images for the model
        """
        self.input_size = input_size
        self.model = None
        self.class_names = None
        
        # Try to load model
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                print(f"Model loaded from {model_path}")
            except:
                model_path = 'models/final_model.h5'
                if os.path.exists(model_path):
                    self.model = keras.models.load_model(model_path)
                    print(f"Model loaded from {model_path}")
                else:
                    print("Warning: No model found. Please train a model first.")
        
        # Initialize class names (for digits 0-9)
        # For full EMNIST with letters, expand this list
        self.class_names = [str(i) for i in range(10)]  # 0-9
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Handwritten Character Recognition")
        self.root.geometry("800x600")
        
        # Drawing canvas
        self.canvas_size = 280
        self.canvas = Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            cursor='crosshair'
        )
        self.canvas.pack(pady=20)
        
        # Bind drawing events
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        
        # Drawing state
        self.last_x = None
        self.last_y = None
        self.line_width = 15
        
        # Control buttons
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)
        
        self.predict_btn = Button(
            button_frame,
            text="Predict",
            command=self.predict,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas,
            bg='#f44336',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Prediction label
        self.prediction_label = Label(
            self.root,
            text="Draw a character and click 'Predict'",
            font=('Arial', 16, 'bold'),
            fg='#333'
        )
        self.prediction_label.pack(pady=20)
        
        # Confidence label
        self.confidence_label = Label(
            self.root,
            text="",
            font=('Arial', 12),
            fg='#666'
        )
        self.confidence_label.pack()
    
    def draw(self, event):
        """Handle drawing on canvas"""
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                width=self.line_width,
                fill='black',
                capstyle=tk.ROUND,
                smooth=tk.TRUE
            )
        self.last_x = event.x
        self.last_y = event.y
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.prediction_label.config(text="Draw a character and click 'Predict'")
        self.confidence_label.config(text="")
    
    def get_canvas_image(self):
        """Convert canvas drawing to numpy array"""
        # Get canvas content as PostScript
        ps = self.canvas.postscript(colormode='mono')
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.convert('L')
        
        # Resize to model input size
        img = img.resize(self.input_size, Image.Resampling.LANCZOS)
        
        # Invert colors (black background to white, white to black)
        img = ImageOps.invert(img)
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype('float32') / 255.0
        
        # Add channel dimension
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self):
        """Predict the drawn character"""
        if self.model is None:
            messagebox.showerror(
                "Error",
                "No model loaded. Please train a model first using train.py"
            )
            return
        
        try:
            # Get image from canvas
            img_array = self.get_canvas_image()
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Update labels
            if self.class_names:
                predicted_char = self.class_names[predicted_class]
            else:
                predicted_char = str(predicted_class)
            
            self.prediction_label.config(
                text=f"Prediction: {predicted_char}",
                fg='#4CAF50'
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2%}"
            )
        
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


# Alternative implementation using PIL directly for better drawing
class HandwritingGUIV2:
    def __init__(self, model_path='models/best_model.h5', input_size=(28, 28)):
        """Improved GUI with better drawing support"""
        self.input_size = input_size
        self.model = None
        self.class_names = [str(i) for i in range(10)]  # 0-9
        
        # Load model
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        else:
            model_path = 'models/final_model.h5'
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Handwritten Character Recognition")
        self.root.geometry("600x700")
        
        # Title
        title_label = Label(
            self.root,
            text="Handwritten Character Recognition",
            font=('Arial', 18, 'bold'),
            pady=10
        )
        title_label.pack()
        
        # Drawing canvas
        self.canvas_size = 280
        self.canvas = Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.canvas.pack(pady=20)
        
        # PIL image for drawing
        self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.pil_image)
        
        # Bind events
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        self.last_x = None
        self.last_y = None
        
        # Buttons
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)
        
        Button(
            button_frame,
            text="Predict",
            command=self.predict,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas,
            bg='#f44336',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Prediction display
        self.prediction_label = Label(
            self.root,
            text="Draw a digit (0-9) and click 'Predict'",
            font=('Arial', 16),
            pady=20
        )
        self.prediction_label.pack()
        
        self.confidence_label = Label(
            self.root,
            text="",
            font=('Arial', 12),
            fg='gray'
        )
        self.confidence_label.pack()
    
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_on_canvas(self, event):
        if self.last_x and self.last_y:
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill='black',
                width=15
            )
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                width=15,
                fill='black',
                capstyle=tk.ROUND
            )
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.pil_image)
        self.prediction_label.config(text="Draw a digit (0-9) and click 'Predict'")
        self.confidence_label.config(text="")
    
    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "No model found. Please train a model first.")
            return
        
        try:
            # Convert PIL image to model input
            img = self.pil_image.convert('L')
            img = img.resize(self.input_size, Image.Resampling.LANCZOS)
            img = ImageOps.invert(img)
            
            img_array = np.array(img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            predicted_char = self.class_names[predicted_class]
            
            self.prediction_label.config(
                text=f"Prediction: {predicted_char}",
                fg='#4CAF50',
                font=('Arial', 18, 'bold')
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2%}",
                font=('Arial', 12)
            )
        
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Use improved GUI
    app = HandwritingGUIV2()
    app.run()

