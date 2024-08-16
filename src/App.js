import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoURL, setVideoURL] = useState('');
  const [audioResults, setAudioResults] = useState([]);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setError('');
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Please upload a video file first.');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/analyze-video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setVideoURL(response.data.video_url);
      setAudioResults(response.data.audio_emotions);
      setLoading(false);
    } catch (err) {
      setError(`Failed to analyze video: ${err.message}`);
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Video Emotion Analysis</h1>
      <input type="file" accept="video/*" onChange={handleFileChange} />
      <button onClick={handleSubmit} disabled={loading || !file}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
      {error && <div className="error">{error}</div>}
      {videoURL && <video src={videoURL} controls style={{ width: '560px', height: '315px' }} />}
      {audioResults.length > 0 && (
        <div>
          <h2>Audio Emotions:</h2>
          <pre>{JSON.stringify(audioResults)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
