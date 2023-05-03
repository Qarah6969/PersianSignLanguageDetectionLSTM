import sys
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QGroupBox, QFrame, QSizePolicy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from celery import Celery

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

app = Celery('tasks', broker = 'pyamqp://guest@localhost//')

class QHSeparationLine(QFrame):
  def __init__(self):
    super().__init__()
    self.setMinimumWidth(1)
    self.setFixedHeight(20)
    self.setFrameShape(QFrame.HLine)
    self.setFrameShadow(QFrame.Sunken)
    self.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Minimum)
    return

@app.task
def predict(sequence, model):
    if len(sequence) == 20:
        res : np.ndarray = model.predict(np.expand_dims(sequence, axis=0))[0]
        return res


class PredictionThread(QThread):
    letter = pyqtSignal(object)

    def __init__(self, parent=None):
        super(PredictionThread, self).__init__(parent)

    def run(self):
        result = predict(self.sequence, self.model)
        self.letter.emit(result)

    def run_task(self, sequence, model):
        self.sequence = sequence
        self.model = model
        self.start()



class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    data = pyqtSignal(object)

    def __init__(self, camera_idx):
        super().__init__()
        self.camera_idx = camera_idx
        self.running = True
        self.translation = False
        self.holistic = mp_holistic.Holistic(model_complexity=0 ,min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def draw_landmarks(self, image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,mp_drawing.DrawingSpec(color=(255,0,255),thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(0,255,255),thickness=1,circle_radius=1))
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    def extract_keys(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh =  np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        face =  np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        return np.concatenate([face, lh, rh])

    def run(self):
        sequence = []
        cap = cv2.VideoCapture(self.camera_idx)
        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.translation:
                    
                    image, results = self.mediapipe_detection(frame, self.holistic)
                    self.draw_landmarks(image, results)
                    keypoints = self.extract_keys(results)
                    sequence.append(keypoints)
                    sequence = sequence[-20:]
                    self.frame_ready.emit(image)
                    self.data.emit(sequence)
                else:
                    self.frame_ready.emit(frame)

        cap.release()

    def stop(self):
        self.running = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.model = None
        self.camera_thread = None
        self.prediction_thread = None
        self.translation_running = False

        main_layout = QVBoxLayout()

        self.camera_group_box = QGroupBox("Camera Feed")
        self.camera_group_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; }")
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_label)
        self.camera_group_box.setLayout(camera_layout)
        main_layout.addWidget(self.camera_group_box)

        self.controls_group_box = QGroupBox("Controls")
        self.controls_group_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; }")
        controls_layout = QVBoxLayout()
        self.load_model_button = QPushButton('Load Model')
        self.load_model_button.setStyleSheet("QPushButton { background-color: #3cba54; color: white; font-weight: bold; font-size: 16px; }"
                                              "QPushButton:disabled { background-color: #bfbfbf; color: black; font-weight: normal; }")
        self.load_model_button.clicked.connect(self.load_model)
        self.camera_combo_box = QComboBox()
        self.populate_camera_combo_box()
        controls_layout.addWidget(self.load_model_button)
        controls_layout.addWidget(QHSeparationLine())
        controls_layout.addWidget(self.camera_combo_box)
        self.camera_button = QPushButton('Start Camera')
        self.camera_button.setStyleSheet("QPushButton { background-color: #3cba54; color: white; font-weight: bold; font-size: 16px; }"
                                          "QPushButton:disabled { background-color: #bfbfbf; color: black; font-weight: normal; }")
        self.camera_button.setEnabled(True)
        self.camera_button.clicked.connect(self.toggle_camera)
        self.translation_button = QPushButton('Start Translation')
        self.translation_button.setStyleSheet("QPushButton { background-color: #3cba54; color: white; font-weight: bold; font-size: 16px; }"
                                               "QPushButton:disabled { background-color: #bfbfbf; color: black; font-weight: normal; }")
        self.translation_button.setEnabled(False)
        self.translation_button.clicked.connect(self.start_translation)
        controls_layout.addWidget(self.camera_button)
        controls_layout.addWidget(QHSeparationLine())
        controls_layout.addWidget(self.translation_button)
        self.translation_label = QLabel()
        self.translation_label.setText("")
        controls_layout.addWidget(self.translation_label)
        self.controls_group_box.setLayout(controls_layout)
        main_layout.addWidget(self.controls_group_box)

        self.setLayout(main_layout)
        self.setWindowTitle('Sign Language Translation')
        self.setFixedSize(800, 750)

    def load_model(self):
        self.load_model_button.setEnabled(False)
        self.actions = np.array(['Alef','B','P','T','Th','Jim','Ch','H','Kh' ,'R'])
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(20,1530)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.actions.shape[0], activation='softmax')
        ])
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.load_weights("./weights.h5")
        self.load_model_button.setEnabled(True)
        self.camera_button.setEnabled(True)
        if self.camera_thread is not None:
            self.translation_button.setEnabled(True)

    def toggle_camera(self):
        if self.camera_thread is None:
            self.start_camera()
            if self.model is not None:
                self.translation_button.setEnabled(True)
        else:
            self.stop_camera()
            self.translation_running = False
            self.translation_button.setEnabled(False)
            self.translation_button.setText("Start Translation")

    def start_camera(self):
        self.camera_button.setText('Stop Camera')
        self.camera_combo_box.setEnabled(False)

        self.camera_thread = CameraThread(int(self.camera_combo_box.currentData()))
        self.camera_thread.frame_ready.connect(self.update_camera)
        self.camera_thread.start()

    def stop_camera(self):
        self.camera_button.setText('Start Camera')
        self.translation_button.setEnabled(False)
        self.camera_combo_box.setEnabled(True)

        self.camera_thread.stop()
        self.camera_thread = None
        

    def update_camera(self, frame):
        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(pixmap)

    def show_predictions(self, res):
        self.translation_label.setText(self.actions[np.argmax(res)])
    
    def start_prediction(self, sequence):
        self.prediction_thread.run_task(sequence, self.model)
        self.prediction_thread.letter.connect(self.show_predictions)

    def start_translation(self):
        if self.translation_running:
            self.translation_button.setText('Start Translation')
            self.translation_running = False
            self.camera_thread.translation = False
            
        else:
            if self.camera_thread is not None and self.model is not None:
                self.translation_button.setText('Stop Translation')
                self.translation_running = True
                self.camera_thread.translation = True
                self.camera_thread.data.connect(self.start_prediction)
                self.prediction_thread = PredictionThread()

    def populate_camera_combo_box(self):

        for j in range(10):
            cap = cv2.VideoCapture(j)
            if cap.isOpened():
                camera_name = f'Camera {j}'
                self.camera_combo_box.addItem(camera_name, j)
                cap.release()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())           
