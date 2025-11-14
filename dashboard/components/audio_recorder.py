"""
Custom audio recorder component for Streamlit.
Uses HTML5 MediaRecorder API.
"""
import streamlit.components.v1 as components

def audio_recorder_component():
    """Create an audio recorder component that returns audio data."""
    audio_html = """
    <div style="padding: 20px; border: 2px solid #6aa36f; border-radius: 10px; background-color: #f0f2f6;">
        <h4 style="margin-top: 0;">🎤 Audio Recorder</h4>
        <button id="startBtn" onclick="startRecording()" style="padding: 12px 24px; font-size: 16px; background-color: #ff4b4b; color: white; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px;">
            🎤 Start Recording
        </button>
        <button id="stopBtn" onclick="stopRecording()" style="padding: 12px 24px; font-size: 16px; background-color: #6aa36f; color: white; border: none; border-radius: 5px; cursor: pointer;" disabled>
            ⏹️ Stop Recording
        </button>
        <div id="status" style="margin-top: 15px; font-weight: bold; color: #666;"></div>
        <div id="audioContainer" style="margin-top: 20px;"></div>
        <div id="downloadLink" style="margin-top: 15px;"></div>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let audioBlob = null;
        
        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = () => {
                        audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = document.createElement('audio');
                        audio.src = audioUrl;
                        audio.controls = true;
                        audio.style.width = '100%';
                        
                        document.getElementById('audioContainer').innerHTML = '';
                        document.getElementById('audioContainer').appendChild(audio);
                        document.getElementById('status').textContent = '✅ Recording complete! Click "Download Audio" below.';
                        
                        // Create download link
                        const downloadBtn = document.createElement('a');
                        downloadBtn.href = audioUrl;
                        downloadBtn.download = 'recording.wav';
                        downloadBtn.textContent = '📥 Download Audio';
                        downloadBtn.style.display = 'inline-block';
                        downloadBtn.style.padding = '10px 20px';
                        downloadBtn.style.backgroundColor = '#1f77b4';
                        downloadBtn.style.color = 'white';
                        downloadBtn.style.textDecoration = 'none';
                        downloadBtn.style.borderRadius = '5px';
                        downloadBtn.style.marginTop = '10px';
                        document.getElementById('downloadLink').innerHTML = '';
                        document.getElementById('downloadLink').appendChild(downloadBtn);
                        
                        // Store blob in window for parent access
                        window.audioBlobUrl = audioUrl;
                        window.audioBlob = audioBlob;
                    };
                    
                    mediaRecorder.start();
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('status').textContent = '🔴 Recording... Click "Stop Recording" when done.';
                })
                .catch(err => {
                    document.getElementById('status').textContent = '❌ Error: ' + err.message;
                    alert('Could not access microphone. Please allow microphone access.');
                });
        }
        
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            }
        }
    </script>
    """
    
    return components.html(audio_html, height=250)

