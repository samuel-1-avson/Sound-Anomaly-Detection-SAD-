import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import socket
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tensorflow.keras.models import load_model
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import time

# Set server details
HOST = '172.20.10.4'  # Replace with your computer's IP address
PORT = 12345  # Replace with the port used by the client

# Path to the pre-trained model
model_path = r'C:\Users\Bridgette Akua Anese\Downloads\Sound-Based-Anomally-Detection-main\Sound-Based-Anomally-Detection-main\autoencoder_model.h5'

# Mailtrap SMTP configuration
MAILTRAP_SMTP_SERVER = 'sandbox.smtp.mailtrap.io'
MAILTRAP_SMTP_PORT = 2525
MAILTRAP_USERNAME = 'f60a8347ee80ba'
MAILTRAP_PASSWORD = 'f32d959494eb8e'
MAILTRAP_FROM_EMAIL = 'samuelavson@gmail.com'
MAILTRAP_TO_EMAIL = 'samuelavson360@gmail.com'

class AudioAnomalyDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Anomaly Detector")
        self.setGeometry(100, 100, 1800, 1200)

        self.init_model()
        self.init_ui()

        # Initialize socket server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(1)
        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"Connection established with {self.client_address}")

        # Initialize anomaly count and threshold
        self.anomaly_count = 0
        self.anomaly_threshold = 100 

    def init_model(self):
        # Load the pre-trained model
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit()

        # Initialize accuracy metrics
        self.accuracy = 0
        self.total_predictions = 0
        self.correct_predictions = 0

    def init_ui(self):
        # Set up the main UI components
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.create_widgets()
        self.create_layout()
        
        # Set up timer for live data updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_live_data)
        
        # Initialize data storage
        self.audio_data = []
        self.sampling_rate = 22050
        self.reconstruction_errors = []
        self.prediction_scores = []
        self.sma_window = 10  # Window size for Simple Moving Average
        self.sma_values = []



    def create_widgets(self):
        # Create labels for displaying information
        self.accuracy_label = QLabel("Model Accuracy: N/A")
        self.accuracy_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.prediction_label = QLabel("Prediction: N/A")
        self.prediction_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self.start_button = QPushButton("Start Live Monitoring")
        self.start_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.start_button.clicked.connect(self.start_live_prediction)
        
        self.stop_button = QPushButton("Stop Live Monitoring")
        self.stop_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.stop_button.clicked.connect(self.stop_live_prediction)
        
        self.send_email_button = QPushButton("Send Email Now")
        self.send_email_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.send_email_button.clicked.connect(self.send_email_now)
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-size: 14px; color: green;")
        
        self.threshold_label = QLabel("Median Threshold: N/A")
        self.threshold_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.create_graphs()

    def create_graphs(self):
        # Create subplots for different visualizations
        self.fig_waveform, self.ax_waveform = plt.subplots(figsize=(5, 3))
        self.canvas_waveform = FigureCanvas(self.fig_waveform)
        
        self.fig_mel, self.ax_mel = plt.subplots(figsize=(5, 3))
        self.canvas_mel = FigureCanvas(self.fig_mel)
        
        self.fig_histogram, self.ax_histogram = plt.subplots(figsize=(5, 3))
        self.canvas_histogram = FigureCanvas(self.fig_histogram)
        
        self.fig_scatter, self.ax_scatter = plt.subplots(figsize=(5, 3))
        self.canvas_scatter = FigureCanvas(self.fig_scatter)

        self.fig_sma, self.ax_sma = plt.subplots(figsize=(5, 3))
        self.canvas_sma = FigureCanvas(self.fig_sma)

    def create_layout(self):
        # Set up the layout for the UI
        graph_layout = QVBoxLayout()
        
        waveform_layout = QHBoxLayout()
        waveform_layout.addWidget(self.canvas_waveform)
        waveform_layout.addWidget(self.canvas_mel)
        graph_layout.addLayout(waveform_layout)
        
        histogram_layout = QHBoxLayout()
        histogram_layout.addWidget(self.canvas_histogram)
        histogram_layout.addWidget(self.canvas_scatter)
        graph_layout.addLayout(histogram_layout)
        
        sma_layout = QHBoxLayout()
        sma_layout.addWidget(self.canvas_sma)
        graph_layout.addLayout(sma_layout)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.send_email_button)
        button_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        self.layout.addWidget(self.accuracy_label)
        self.layout.addWidget(self.prediction_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.threshold_label)
        self.layout.addLayout(button_layout)
        self.layout.addLayout(graph_layout)

    def start_live_prediction(self):
        self.status_label.setText("Status: Monitoring")
        self.status_label.setStyleSheet("font-size: 14px; color: orange;")
        self.timer.start(1000)  # Update every 1 second

    def stop_live_prediction(self):
        self.status_label.setText("Status: Idle")
        self.status_label.setStyleSheet("font-size: 14px; color: green;")
        self.timer.stop()

    def calculate_sma(self, data, window):
        # Calculate Simple Moving Average
        if len(data) < window:
            return np.array([])
        return np.convolve(data, np.ones(window), 'valid') / window

    def calculate_median_threshold(self):
        # Calculate the median threshold from reconstruction errors
        if len(self.reconstruction_errors) > 0:
            return np.median(self.reconstruction_errors)
        return 0  # Default value if no data is available
    
    def calculate_ema(self, data, window):
        # Calculate Exponential Moving Average
        ema = np.zeros_like(data)
        alpha = 2 / (window + 1)
        ema[0] = data[0]  # Initialize EMA with the first data point
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def update_live_data(self):
        try:
            # Receive audio data from the client
            audio_length = int(self.sampling_rate * 1)  # 1 second worth of data
            audio_data = self.client_socket.recv(audio_length * 4)  # 4 bytes per float32
            if len(audio_data) == 0:
                return
            self.audio_data = np.frombuffer(audio_data, dtype=np.float32)

            if len(self.audio_data) == 0:
                return

            y = np.array(self.audio_data, dtype=np.float32)

            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=self.sampling_rate, n_mels=128, n_fft=1024, hop_length=512)
            S_db = librosa.power_to_db(S, ref=np.max)

            # Prepare input data for the model
            target_length = 2048
            if S_db.size < target_length:
                pad_width = target_length - S_db.size
                input_data = np.pad(S_db.flatten(), (0, pad_width), mode='constant')
            else:
                input_data = S_db.flatten()[:target_length]

            input_data = input_data.reshape(1, -1)

            if input_data.shape[1] != 2048:
                print(f"Error: reshaped input size {input_data.shape[1]} does not match expected size 2048")
                return

            # Make prediction using the model
            prediction = self.model.predict(input_data)

            # Calculate reconstruction error
            reconstruction_error = np.mean(np.square(input_data - prediction))
            self.reconstruction_errors.append(reconstruction_error)
            
            # Calculate median threshold
            self.reconstruction_threshold = self.calculate_median_threshold()
            
            # Determine if it's an anomaly based on reconstruction error and median threshold
            is_anomaly = reconstruction_error > self.reconstruction_threshold
            
            prediction_str = 'Anomaly' if is_anomaly else 'Normal'
            self.prediction_label.setText(f"Prediction: {prediction_str}")
            self.prediction_scores.append(reconstruction_error)
            
            # Update threshold label
            self.threshold_label.setText(f"Median Threshold: {self.reconstruction_threshold:.4f}")

            # Send prediction and threshold to client
            data_to_send = f"{prediction_str},{self.reconstruction_threshold:.4f}"
            self.client_socket.sendall(data_to_send.encode())

            # Calculate SMA
            self.sma_values = self.calculate_sma(self.reconstruction_errors, self.sma_window)

            # Update visualizations
            self.update_waveform(y)
            self.update_mel_spectrogram(S_db)
            self.update_histogram()
            self.update_scatter_plot()
            self.update_sma_plot()
            self.update_accuracy(is_anomaly)

            # Increment anomaly count and check threshold
            if is_anomaly:
                self.anomaly_count += 1
                if self.anomaly_count > self.anomaly_threshold:
                    self.send_email("Anomaly Alert", f"Anomalies have been detected {self.anomaly_count} times, exceeding the threshold of {self.anomaly_threshold}. Please check the attached graphs.")
                    self.anomaly_count = 0  # Reset the count after sending the email

        except Exception as e:
            print(f"Error during live data update: {e}")



    def update_waveform(self, y):
        # Update waveform plot
        self.ax_waveform.clear()
        self.ax_waveform.plot(y)
        self.ax_waveform.set_title("Waveform")
        self.ax_waveform.set_xlabel("Time")
        self.ax_waveform.set_ylabel("Amplitude")
        self.canvas_waveform.draw()

    def update_mel_spectrogram(self, S_db):
        # Update mel spectrogram plot
        self.ax_mel.clear()
        librosa.display.specshow(S_db, sr=self.sampling_rate, x_axis='time', y_axis='mel', ax=self.ax_mel)
        self.ax_mel.set_title("Mel Spectrogram")
        self.canvas_mel.draw()

    def update_histogram(self):
        # Update histogram of reconstruction errors
        self.ax_histogram.clear()
        self.ax_histogram.hist(self.reconstruction_errors, bins=30, color='purple')
        self.ax_histogram.set_title("Histogram of Reconstruction Errors")
        self.ax_histogram.set_xlabel("Reconstruction Error")
        self.ax_histogram.set_ylabel("Frequency")
        self.canvas_histogram.draw()

    def update_scatter_plot(self):
        # Update scatter plot of reconstruction errors
        self.ax_scatter.clear()
        if len(self.prediction_scores) > 1:
            self.ax_scatter.scatter(range(len(self.prediction_scores)), self.prediction_scores, color='green', alpha=0.5)
            self.ax_scatter.set_title("Scatter Plot of Reconstruction Errors")
            self.ax_scatter.set_xlabel("Index")
            self.ax_scatter.set_ylabel("Reconstruction Error")
        else:
            self.ax_scatter.set_title("Scatter Plot of Reconstruction Errors (Not Enough Data)")
        self.canvas_scatter.draw()

    def update_sma_plot(self):
        # Update Simple Moving Average plot
        self.ax_sma.clear()
        if len(self.reconstruction_errors) > self.sma_window:
            x = range(self.sma_window - 1, len(self.reconstruction_errors))
            self.ax_sma.plot(x, self.reconstruction_errors[self.sma_window-1:], label='Reconstruction Error', alpha=0.5)

            # Plot SMA
            self.ax_sma.plot(x, self.sma_values, label=f'SMA ({self.sma_window})', color='red')
        
            # Calculate and plot EMA
            ema_values = self.calculate_ema(self.reconstruction_errors, self.sma_window)
            self.ax_sma.plot(x, ema_values[self.sma_window-1:], label=f'EMA ({self.sma_window})', color='blue')

            # Plot median threshold
            self.ax_sma.axhline(y=self.reconstruction_threshold, color='g', linestyle='--', label='Median Threshold')

            self.ax_sma.set_title("SMA and EMA of Reconstruction Errors")
            self.ax_sma.set_xlabel("Index")
            self.ax_sma.set_ylabel("Error")
            self.ax_sma.legend()
        else:
            self.ax_sma.set_title("Simple Moving Average (Not Enough Data)")
        self.canvas_sma.draw()

    def update_accuracy(self, is_anomaly):
        # Update accuracy metrics (using simulated ground truth)
        self.total_predictions += 1
        ground_truth = 1 if np.random.random() < 0.2 else 0  # 20% chance of being an anomaly
        if is_anomaly == ground_truth:
            self.correct_predictions += 1
        
        self.accuracy = self.correct_predictions / self.total_predictions
        self.accuracy_label.setText(f"Model Accuracy: {self.accuracy:.2%}")

    def send_email(self, subject, body):
        try:
            msg = MIMEMultipart()
            msg['From'] = MAILTRAP_FROM_EMAIL
            msg['To'] = MAILTRAP_TO_EMAIL
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

        # Save and attach figures
            temp_dir = self.save_figures()
            self.attach_images(msg, temp_dir)

            with smtplib.SMTP(MAILTRAP_SMTP_SERVER, MAILTRAP_SMTP_PORT) as server:
                server.set_debuglevel(1)  # Enable debug output
                server.starttls()
                server.login(MAILTRAP_USERNAME, MAILTRAP_PASSWORD)
                server.sendmail(MAILTRAP_FROM_EMAIL, MAILTRAP_TO_EMAIL, msg.as_string())
            print("Email sent successfully.")
        except smtplib.SMTPException as e:
            print(f"SMTP error occurred: {e}")
        except Exception as e:
            print(f"Failed to send email: {e}")
        finally:
            # Clean up temporary images
            for filename in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, filename))
            os.rmdir(temp_dir)



    def send_email_now(self):
    # Send email with a predefined subject and body
        subject = "Manual Email Triggered"
        body = "This is a manually triggered email from the Audio Anomaly Detector application with attached graphs."
        self.send_email(subject, body)

    def save_figures(self):
    # Save the figures as images in a temporary directory
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

         # Save each figure
        self.fig_waveform.savefig(os.path.join(temp_dir, 'waveform.png'))
        self.fig_mel.savefig(os.path.join(temp_dir, 'mel_spectrogram.png'))
        self.fig_histogram.savefig(os.path.join(temp_dir, 'histogram.png'))
        self.fig_scatter.savefig(os.path.join(temp_dir, 'scatter_plot.png'))
        self.fig_sma.savefig(os.path.join(temp_dir, 'sma.png'))
        return temp_dir
    
    def attach_images(self, msg, directory):
    # Attach each image to the email
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                img = MIMEBase('application', 'octet-stream')
                img.set_payload(f.read())
                encoders.encode_base64(img)
                img.add_header('Content-Disposition', f'attachment; filename={filename}')
                msg.attach(img)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioAnomalyDetectorGUI()
    window.show()
    sys.exit(app.exec_())
