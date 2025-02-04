from flask import Flask, request, jsonify
import io
import os
import whisper
import librosa
import torch
from soundfile import SoundFile
# from TTS.api import TTS
# from TTS.vocoder import Vocoder

# Initialize the model and app for the entire session (model is thread safe)
device = "cpu" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(name="large", device=device, in_memory=True)
# spectrogram = TTS(model_name="./models/tacotron2", progress_bar=False)
# vocoder = Vocoder.init_vocoder_from_path("./models/HiFi-GAN")
app = Flask(__name__)

# Receives audio files and replies with a transcript
@app.route('/upload', methods=['POST'])
def handle_upload():
    try:
        # Convert the file to a tensor, avoiding SSD-killing file writes
        input = io.BytesIO(request.data)
        data, _ = librosa.load(SoundFile(input))
        tensor = torch.tensor(data)

        # Convert the file to text using whisper
        result = model.transcribe(tensor)

        # Reply with the recognized text
        with open("output.txt", "w") as file:
            file.write(result["text"])
        return result["text"], 200
    except Exception as _:
        return jsonify({"error": "Invalid file."}), 422

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3030)