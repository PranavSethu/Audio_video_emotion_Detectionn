from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import librosa
import utils
import os

app = Flask(__name__, static_folder='temp')
CORS(app)

print("Loading models...")
face_model = load_model('model.h5', compile=False)
audio_model = load_model('audio_emotion_detection.h5', compile=False)
print("Models loaded successfully.")

def predict_emotion(face, model):
    if face is None:
        return 'No face detected'
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255
    face = np.expand_dims(face, axis=0)
    face is np.expand_dims(face, axis=-1)
    prediction = model.predict(face)
    return int(np.argmax(prediction))

def process_video(filepath, model):
    cap = cv2.VideoCapture(filepath)
    emotions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        face = utils.extract_face(frame)
        if face is None:
            emotions.append('No face detected')
        else:
            emotion = predict_emotion(face, model)
            emotions.append(emotion)
    cap.release()
    return emotions

def overlay_emotions_on_video(input_path, output_path, emotions):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from 'mp4v' to 'avc1'
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        emotion = str(emotions[frame_count]) if frame_count < len(emotions) else "No face detected"
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()



@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    try:
        file = request.files['file']
        save_dir = './temp'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, file.filename)
        result_video_path = os.path.join(save_dir, "result_" + file.filename)
        file.save(filepath)

        video_emotions = process_video(filepath, face_model)
        overlay_emotions_on_video(filepath, result_video_path, video_emotions)
        audio_features = utils.extract_audio_features(filepath)
        if audio_features is not None:
            audio_emotions = audio_model.predict(audio_features)
            audio_emotions = [int(emotion) for emotion in np.argmax(audio_emotions, axis=1)]
        else:
            audio_emotions = []

        return jsonify({
            'video_url': request.host_url.rstrip('/') + '/temp/' + os.path.basename(result_video_path),
            'audio_emotions': audio_emotions
        })
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/temp/<filename>')
def send_file(filename):
    return send_from_directory(app.static_folder, filename, as_attachment=True, mimetype='video/mp4')


if __name__ == '__main__':
    print("Starting server...")
    app.run(debug=True, use_reloader=False)
