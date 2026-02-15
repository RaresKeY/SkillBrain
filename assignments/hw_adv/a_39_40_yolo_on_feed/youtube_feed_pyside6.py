import sys
import cv2
import yt_dlp
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import QThread, Signal, Qt, Slot
from PySide6.QtGui import QImage, QPixmap

class VideoThread(QThread):
    change_pixmap_signal = Signal(QImage)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self._run_flag = True

    def run(self):
        # Extract direct stream URL using yt-dlp
        ydl_opts = {'format': 'best', 'quiet': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                stream_url = info['url']
        except Exception as e:
            print(f"Error extracting stream: {e}")
            return

        cap = cv2.VideoCapture(stream_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Fallback for streams with unknown FPS
        delay = int(1000 / fps)

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                # Convert BGR to RGB for PySide6
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                height, width, channel = cv_img.shape
                bytes_per_line = channel * width
                qt_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img.copy())
                
                # Wait to match FPS
                self.msleep(delay)
            else:
                # If stream fails, wait a bit and try to reconnect (basic error handling)
                cap.release()
                self.msleep(2000)
                cap = cv2.VideoCapture(stream_url)
                
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Resizable Feed")
        self.resize(1024, 768)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        
        # Allow the label to scale down
        self.image_label.setMinimumSize(1, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.current_qimage = None
        
        # Start the video thread
        self.thread = VideoThread("https://www.youtube.com/watch?v=809RMryNnlM")
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @Slot(QImage)
    def update_image(self, qt_img):
        self.current_qimage = qt_img
        self.display_image()

    def display_image(self):
        if self.current_qimage:
            # Scale the image to fit the label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(self.current_qimage).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-display the image to match new size immediately
        self.display_image()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
