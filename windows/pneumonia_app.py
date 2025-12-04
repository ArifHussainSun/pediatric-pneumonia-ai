#!/usr/bin/env python3
"""
Pneumonia Detection Desktop Application

Windows desktop application for pneumonia detection using MobileNet models.
Optimized for Surface Pro tablets with touch interface.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QFileDialog, QTextEdit,
                             QMessageBox, QProgressBar, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon

from inference_engine import ONNXInferenceEngine


class PneumoniaDetectionApp(QMainWindow):
    """
    Main application window for pneumonia detection.

    Provides touch-friendly interface for Windows tablets.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pneumonia Detection - Windows Desktop")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize UI
        self.init_ui()

        # Initialize inference engine
        self.inference_engine = None
        self.current_image_path = None

        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Try to load default model
        self.load_default_model()

        # Setup auto-save timer for crash recovery
        self.setup_auto_recovery()

        # Restore previous session
        self.restore_session_state()

    def init_ui(self):
        """Initialize the user interface."""

        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                min-height: 60px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left panel for image display
        left_panel = self.create_image_panel()
        main_layout.addWidget(left_panel, 2)

        # Right panel for controls and results
        right_panel = self.create_control_panel()
        main_layout.addWidget(right_panel, 1)

    def create_image_panel(self):
        """Create the image display panel."""

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Chest X-Ray Image")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #BDBDBD;
                border-radius: 10px;
                background-color: white;
                color: #757575;
                font-size: 16px;
            }
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No image selected\n\nClick 'Select Image' to load a chest X-ray")
        layout.addWidget(self.image_label)

        return panel

    def create_control_panel(self):
        """Create the control and results panel."""

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)

        # Title
        title = QLabel("Pneumonia Detection")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Select image button
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        # Analyze button
        self.analyze_button = QPushButton("Analyze X-Ray")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)

        # Results area
        results_label = QLabel("Results:")
        results_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(results_label)

        self.results_text = QTextEdit()
        self.results_text.setMinimumHeight(300)
        self.results_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #BDBDBD;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
                background-color: white;
            }
        """)
        self.results_text.setPlaceholderText("Analysis results will appear here...")
        layout.addWidget(self.results_text)

        # Add stretch to push everything to top
        layout.addStretch()

        return panel

    def select_image(self):
        """Open file dialog to select an image."""

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Chest X-Ray Image",
                "",
                "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp);;All Files (*)"
            )

            if file_path:
                # Validate file size (max 50MB)
                if Path(file_path).stat().st_size > 50 * 1024 * 1024:
                    self.show_warning("File too large", "Please select an image smaller than 50MB.")
                    return

                # Validate file extension
                valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
                if Path(file_path).suffix.lower() not in valid_extensions:
                    self.show_warning("Invalid file type", "Please select a valid image file.")
                    return

                self.load_image(file_path)

        except Exception as e:
            self.show_error(f"Failed to select image: {str(e)}")

    def load_image(self, file_path):
        """Load and display the selected image."""

        try:
            # Load and scale image
            pixmap = QPixmap(file_path)

            if pixmap.isNull():
                self.show_error("Failed to load image. Please select a valid image file.")
                return

            # Scale image to fit display
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)
            self.current_image_path = file_path

            # Enable analyze button
            self.analyze_button.setEnabled(True)

            # Update results with image info
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode

                size_mb = Path(file_path).stat().st_size / (1024 * 1024)

                self.results_text.setText(f"""Image loaded: {Path(file_path).name}

Image Information:
• Resolution: {width} × {height}
• Color mode: {mode}
• File size: {size_mb:.1f} MB

✅ Ready for pneumonia analysis.""")
            except:
                self.results_text.setText(f"Image loaded: {Path(file_path).name}\n\nReady for analysis.")

            self.status_bar.showMessage(f"Image loaded: {Path(file_path).name}", 3000)

        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")

    def analyze_image(self):
        """Analyze the loaded image for pneumonia."""

        if not self.current_image_path:
            self.show_error("No image selected.")
            return

        if not self.inference_engine:
            self.show_error("Model not loaded. Please check if model file exists.")
            return

        try:
            # Disable analyze button during processing
            self.analyze_button.setEnabled(False)
            self.analyze_button.setText("Analyzing...")

            # Update status
            self.results_text.setText("Running pneumonia detection...")

            # Run inference
            result = self.inference_engine.predict(self.current_image_path)

            # Display results
            self.display_results(result)

        except Exception as e:
            self.show_error(f"Analysis failed: {str(e)}")

        finally:
            # Re-enable analyze button
            self.analyze_button.setEnabled(True)
            self.analyze_button.setText("Analyze X-Ray")

    def display_results(self, result):
        """Display analysis results."""

        prediction = result['prediction']
        confidence = result['confidence']
        probabilities = result['probabilities']
        inference_time = result['inference_time_ms']

        # Format results
        results_text = f"""DIAGNOSIS: {prediction}
Confidence: {confidence:.1%}

Detailed Probabilities:
• Normal: {probabilities['NORMAL']:.1%}
• Pneumonia: {probabilities['PNEUMONIA']:.1%}

Performance:
• Inference Time: {inference_time:.0f} ms
• Model: ONNX Runtime

Image: {Path(self.current_image_path).name}
"""

        # Add image quality information if available
        if result.get('image_quality'):
            quality = result['image_quality']
            results_text += f"""
Image Quality Assessment:
• Overall Quality: {quality['overall_quality'].title()}
• Brightness: {quality['brightness_score']:.2f}
• Contrast: {quality['contrast_score']:.2f}
• Sharpness: {quality['sharpness_score']:.2f}
• Noise Level: {quality['noise_level']:.2f}
• Enhancement Applied: {'Yes' if quality['enhancement_applied'] else 'No'}
"""

            if quality['recommendations']:
                results_text += f"\nRecommendations:\n"
                for rec in quality['recommendations']:
                    results_text += f"• {rec}\n"

        self.results_text.setText(results_text)

        # Show user feedback if available
        if result.get('user_feedback'):
            self.show_warning("Image Quality Notice", result['user_feedback'])

        # Pneumonia warning disabled per user request

    def load_default_model(self):
        """Try to load default ONNX model."""

        model_paths = [
            "windows_exports/mobilenet_windows.onnx",
            "../windows_exports/mobilenet_windows.onnx",
            "mobilenet_windows.onnx"
        ]

        for model_path in model_paths:
            if Path(model_path).exists():
                try:
                    self.inference_engine = ONNXInferenceEngine(model_path)
                    self.results_text.setText(f"Model loaded: {model_path}\n\nReady for analysis.")
                    return
                except Exception as e:
                    continue

        # No model found
        self.results_text.setText("""No model found.

To use this application:
1. Export model: python scripts/export_windows.py --model_path your_model.pth
2. Restart application

The app will look for: mobilenet_windows.onnx""")

    def setup_auto_recovery(self):
        """Setup auto-recovery system."""
        self.recovery_timer = QTimer()
        self.recovery_timer.timeout.connect(self.save_session_state)
        self.recovery_timer.start(30000)  # Save every 30 seconds

    def save_session_state(self):
        """Save current session state for crash recovery."""
        try:
            if self.current_image_path:
                with open(".session_recovery", "w") as f:
                    f.write(self.current_image_path)
        except:
            pass  # Silent fail for recovery

    def restore_session_state(self):
        """Restore previous session if available."""
        try:
            if Path(".session_recovery").exists():
                with open(".session_recovery", "r") as f:
                    image_path = f.read().strip()
                    if Path(image_path).exists():
                        self.load_image(image_path)
                        self.status_bar.showMessage("Session restored", 3000)
        except:
            pass  # Silent fail for recovery

    def show_error(self, message):
        """Display error message with popup and status."""
        self.results_text.setText(f"❌ Error: {message}")
        self.status_bar.showMessage(f"Error: {message}", 5000)

        # Also show popup for critical errors
        if "failed" in message.lower() or "crashed" in message.lower():
            QMessageBox.critical(self, "Error", message)

    def show_warning(self, title, message):
        """Display warning message."""
        QMessageBox.warning(self, title, message)
        self.status_bar.showMessage(f"Warning: {message}", 3000)

    def show_info(self, title, message):
        """Display information message."""
        QMessageBox.information(self, title, message)

    def closeEvent(self, event):
        """Handle application closing."""
        try:
            # Clean up recovery file
            if Path(".session_recovery").exists():
                Path(".session_recovery").unlink()

            # Clean up inference engine
            if self.inference_engine:
                del self.inference_engine

            event.accept()
        except:
            event.accept()  # Always allow closing


def main():
    """Main application entry point."""

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Pneumonia Detection")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Pediatric AI Research")

    # Create and show main window
    window = PneumoniaDetectionApp()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()