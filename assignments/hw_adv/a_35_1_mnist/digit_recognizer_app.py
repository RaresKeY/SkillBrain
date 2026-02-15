import sys
import os
import numpy as np
import tensorflow as tf
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFrame, QComboBox, QScrollArea)
from PySide6.QtGui import QPainter, QPen, QImage, QColor, QFont, QBrush
from PySide6.QtCore import Qt, QPoint, QTimer, QRectF
from PIL import Image

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Disable GPU to avoid CUDA errors on systems without configured GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#f0f0f0')
        super(MplCanvas, self).__init__(self.fig)

class DrawingWidget(QWidget):
    def __init__(self, parent=None, on_change_callback=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)  # 10x scale of 28x28
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.last_point = QPoint()
        self.drawing = False
        self.on_change_callback = on_change_callback
        
        # Debounce timer for prediction
        self.prediction_timer = QTimer()
        self.prediction_timer.setSingleShot(True)
        self.prediction_timer.setInterval(500)  # 500ms delay
        if on_change_callback:
            self.prediction_timer.timeout.connect(on_change_callback)
        
        # Pen settings
        self.pen_width = 20
        self.pen_color = Qt.white

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()
            self.draw_point(event.position().toPoint())
            self.prediction_timer.stop() # Stop timer while drawing

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            self.draw_line_to(event.position().toPoint())
            self.prediction_timer.stop() # Reset timer on move

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            # Start timer when user stops drawing
            self.prediction_timer.start()

    def draw_point(self, point):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPoint(point)
        self.update()

    def draw_line_to(self, end_point):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.last_point, end_point)
        self.last_point = end_point
        self.update()

    def clear(self):
        self.image.fill(Qt.black)
        self.update()
        self.prediction_timer.stop() # Cancel any pending prediction
        if self.on_change_callback:
            pass

    def get_image_data(self):
        """
        Returns the image as a (1, 28, 28) normalized numpy array suitable for the model.
        """
        # 1. Resize to 28x28 using Qt's scaling for high quality downsampling
        scaled_image = self.image.scaled(28, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        
        # 2. Convert to Grayscale
        grayscale_image = scaled_image.convertToFormat(QImage.Format_Grayscale8)
        
        # 3. Convert to Numpy Array
        width = grayscale_image.width()
        height = grayscale_image.height()
        
        ptr = grayscale_image.bits()
        # PySide6 bits() returns a memoryview. 
        # We must handle potential padding (bytesPerLine).
        stride = grayscale_image.bytesPerLine()
        
        # Create a copy of the data to avoid lifetime issues
        arr = np.array(ptr, copy=True).reshape(height, stride)
        
        # Crop to the actual width (removes padding if any)
        arr = arr[:, :width]
        
        # 4. Normalize (0-255 -> 0.0-1.0)
        normalized = arr.astype(np.float32) / 255.0
        
        # 5. Reshape for Model (1, 28, 28)
        return normalized.reshape(1, 28, 28)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Digit Recognizer")
        self.setGeometry(100, 100, 1000, 600) # Wide window for side-by-side
        
        # Setup debug dir
        self.debug_dir = os.path.join(os.path.dirname(__file__), 'debug_images')
        os.makedirs(self.debug_dir, exist_ok=True)
        self.prediction_count = 0
        
        # Main Layout (Left vs Right)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- LEFT PANEL (Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(320)
        
        # Model Selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Dense Model", "CNN Model"])
        self.model_selector.currentIndexChanged.connect(self.change_model)
        model_layout.addWidget(self.model_selector)
        left_layout.addLayout(model_layout)
        
        # Instruction
        instruction_label = QLabel("Draw a digit (0-9)")
        instruction_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(instruction_label)
        
        # Drawing Area
        draw_container = QFrame()
        draw_layout = QHBoxLayout(draw_container)
        draw_layout.setContentsMargins(0,0,0,0)
        
        self.drawer = DrawingWidget(on_change_callback=self.predict_digit)
        self.drawer.setStyleSheet("border: 2px solid #555;")
        
        draw_layout.addStretch()
        draw_layout.addWidget(self.drawer)
        draw_layout.addStretch()
        left_layout.addWidget(draw_container)
        
        # Prediction Result
        self.result_label = QLabel("Draw to predict")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 24, QFont.Bold))
        left_layout.addWidget(self.result_label)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.confidence_label)
        
        # Clear Button
        clear_btn = QPushButton("Clear")
        clear_btn.setMinimumHeight(40)
        clear_btn.clicked.connect(self.clear_all)
        left_layout.addWidget(clear_btn)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # --- RIGHT PANEL (Visualizations) ---
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        
        self.sc = MplCanvas(self, width=6, height=8, dpi=100)
        self.right_layout.addWidget(self.sc)
        
        main_layout.addWidget(right_panel)

        # Initial Load
        self.model = None
        self.activation_model = None
        self.bars = [] # To store bar containers for updating
        self.change_model(0)

    def change_model(self, index):
        model_name = self.model_selector.currentText()
        filename = 'mnist_model.keras' if model_name == "Dense Model" else 'mnist_cnn_model.keras'
        self.load_model(filename)
        # Re-predict if there is a drawing
        self.predict_digit()

    def load_model(self, filename):
        try:
            model_path = os.path.join(os.path.dirname(__file__), filename)
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Please run training script first.")
                self.model = None
                self.activation_model = None
                self.result_label.setText("Model not found")
                return
            
            print(f"Loading model from {model_path}...")
            self.model = tf.keras.models.load_model(model_path)
            
            # Identify Dense Layers and Output
            self.layer_names = []
            layer_outputs = []
            
            # Robustly find inputs and outputs
            # Try getting input shape
            input_shape = None
            try:
                input_shape = self.model.input_shape
            except AttributeError:
                 # Fallback for Sequential without defined input
                 # We can inspect the first layer
                 first_layer = self.model.layers[0]
                 if hasattr(first_layer, 'input_shape'):
                     input_shape = first_layer.input_shape
            
            if input_shape:
                 # input_shape tuple usually includes batch dim (None, 28, 28)
                 # tf.keras.Input expects shape WITHOUT batch dim
                 shape_no_batch = input_shape[1:] 
                 new_input = tf.keras.Input(shape=shape_no_batch)
                 
                 # Reconstruct graph
                 x = new_input
                 for layer in self.model.layers:
                     x = layer(x)
                     if isinstance(layer, tf.keras.layers.Dense):
                         layer_outputs.append(x)
                         self.layer_names.append(layer.name)
                 
                 if layer_outputs:
                     self.activation_model = tf.keras.Model(inputs=new_input, outputs=layer_outputs)
            
            # Fallback if reconstruction failed (e.g. complex model)
            if not self.activation_model:
                print("Could not reconstruct model for visualization. Falling back to standard prediction.")
                # We will just predict normally in predict_digit if activation_model is None
            
            # Setup Plots
            self.setup_plots()
            
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.activation_model = None
            self.result_label.setText("Error loading model")

    def setup_plots(self):
        self.sc.fig.clear()
        self.bars = []
        
        if not self.activation_model:
            self.sc.draw()
            return
            
        num_layers = len(self.layer_names)
        if num_layers == 0: return
        
        # Create subplots
        self.axes = self.sc.fig.subplots(num_layers, 1)
        if num_layers == 1: self.axes = [self.axes]
        
        self.sc.fig.subplots_adjust(hspace=0.5)
        
        for i, ax in enumerate(self.axes):
            # Placeholder data
            shape = self.activation_model.outputs[i].shape
            size = shape[-1]
            
            data = np.zeros(size)
            x = np.arange(size)
            
            rects = ax.bar(x, data, color='blue', width=0.8)
            self.bars.append(rects)
            
            ax.set_title(f"Layer: {self.layer_names[i]} ({size} neurons)")
            ax.set_ylim(0, 1.1) # Activations usually 0-1 or ReLU max
            
            if i == num_layers - 1:
                # Output layer
                ax.set_xticks(range(10))
                ax.set_ylim(0, 1.0)
            else:
                ax.set_xticks([]) # Hide x ticks for hidden layers to reduce clutter
                
        self.sc.draw()

    def clear_all(self):
        self.drawer.clear()
        self.result_label.setText("Draw to predict")
        self.confidence_label.setText("")
        # Reset charts
        for rects in self.bars:
            for rect in rects:
                rect.set_height(0)
        self.sc.draw()

    def save_debug_image(self, image_data):
        """Saves the input image to debug folder, keeping only last 5."""
        try:
            # image_data is (1, 28, 28) float 0-1
            img_array = (image_data[0] * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            filename = f"pred_{self.prediction_count % 5}.png"
            path = os.path.join(self.debug_dir, filename)
            img.save(path)
            self.prediction_count += 1
            print(f"Saved debug image to {path}")
        except Exception as e:
            print(f"Failed to save debug image: {e}")

    def predict_digit(self):
        if self.model is None:
            return
            
        input_data = self.drawer.get_image_data()
        self.save_debug_image(input_data)
        
        if np.max(input_data) < 0.1:
            self.clear_all()
            return

        input_shape = self.model.input_shape
        # Handle input shape flexibility (e.g. if we reconstructed it slightly differently)
        # But generally:
        if len(input_shape) == 4: # CNN
             model_input = np.expand_dims(input_data, axis=-1)
        else: # Dense
             model_input = input_data

        if self.activation_model:
            # Get all activations
            outputs = self.activation_model(model_input, training=False)
            if not isinstance(outputs, list):
                outputs = [outputs]
            
            # Update plots
            for i, output in enumerate(outputs):
                vals = output.numpy()[0]
                
                # For output layer, softmax it if it's logits
                if i == len(outputs) - 1:
                    vals = tf.nn.softmax(vals).numpy()
                
                # Update bars
                rects = self.bars[i]
                
                # Determine max for scaling y-axis dynamically if needed
                max_val = np.max(vals)
                if max_val > self.axes[i].get_ylim()[1]:
                     self.axes[i].set_ylim(0, max_val * 1.1)
                
                for j, rect in enumerate(rects):
                    rect.set_height(vals[j])
                    if i == len(outputs) - 1:
                         if vals[j] > 0.5:
                             rect.set_color('green')
                         else:
                             rect.set_color('blue')

            self.sc.draw()
            
            # Update labels from last layer output
            final_probs = tf.nn.softmax(outputs[-1]).numpy()[0]
            predicted_digit = np.argmax(final_probs)
            confidence = np.max(final_probs)
            
            self.result_label.setText(f"Prediction: {predicted_digit}")
            self.confidence_label.setText(f"Confidence: {confidence:.2%}")

        else:
            # Fallback
            logits = self.model(model_input, training=False)
            probs = tf.nn.softmax(logits).numpy()[0]
            predicted_digit = np.argmax(probs)
            confidence = np.max(probs)
            self.result_label.setText(f"Prediction: {predicted_digit}")
            self.confidence_label.setText(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())