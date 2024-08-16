import cv2
import numpy as np
import librosa
import moviepy.editor as mp
from flask import current_app as app

# Initialize the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(frame):
    try:
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        return face
    except Exception as e:
        app.logger.error(f"Error extracting face: {str(e)}")
        return None

def extract_audio_features(video_path):
    try:
        clip = mp.VideoFileClip(video_path)
        audio = clip.audio
        audio_path = 'temp_audio.wav'
        audio.write_audiofile(audio_path, codec='pcm_s16le')  # Ensure codec compatibility
        clip.close()  # Close the video file to free up system resources
    except Exception as e:
        app.logger.error(f"Failed to extract audio from video: {str(e)}")
        return None

    try:
        y, sr = librosa.load(audio_path, sr=None)  # Load audio without changing the sample rate
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCCs to match model's expectation
        mfccs_processed = np.mean(mfccs.T, axis=0)
        # Reshape to match the input shape of the model: (None, 13, 1, 1)
        mfccs_processed = mfccs_processed.reshape(1, 13, 1, 1)
        return mfccs_processed
    except Exception as e:
        app.logger.error(f"Failed to process audio features: {str(e)}")
        return None

