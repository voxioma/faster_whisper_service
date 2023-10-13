from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
import time
import uuid
import os
import subprocess

model_size = "large-v2"
model = WhisperModel(model_size, device="cuda", compute_type="int8")

app = Flask(__name__)

def generate_random_filename(extension=".txt"):
    """Generate a random filename with the given extension."""
    return f"{uuid.uuid4()}{extension}"

def preprocess_audio(input_filename):
    """Convert audio to 16kHz sample rate and mono channel using FFmpeg."""
    output_filename = generate_random_filename(extension=".wav")
    command = [
        'ffmpeg',
        '-i', input_filename,
        '-ac', '1',
        '-ar', '16000',
        output_filename
    ]
    subprocess.run(command, check=True)
    return output_filename

@app.route('/transcribe', methods=['POST'])
async def transcribe():

    # Get the file from the request
    file = request.files['inputFile']

    # Generate a random filename and save the uploaded file
    fn = generate_random_filename(extension=".wav")
    file.save(fn)

    preprocessed_fn = preprocess_audio(fn)

    segments, info = model.transcribe(preprocessed_fn, beam_size=5)

    start = time.time()

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    total_text = ""

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        total_text += segment.text + ". "

    print("\nTIME TAKEN: " + str(time.time() - start) + "\n")

    return jsonify({"transcription": total_text})


if __name__ == '__main__':
    app.run(host="10.128.0.13",debug=True, port=2424, use_reloader=False)
