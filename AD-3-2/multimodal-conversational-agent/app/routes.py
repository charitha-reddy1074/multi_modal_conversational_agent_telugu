from flask import Blueprint, request, render_template, redirect, url_for
from models.image_model import process_image
from models.audio_model import process_audio, audio_to_text_and_speech
from models.text_model import process_text
import os

app_routes = Blueprint('app_routes', __name__)

@app_routes.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app_routes.route('/process', methods=['POST'])
def process_input():
    image_file = request.files.get('image')
    audio_file = request.files.get('audio')
    text_input = request.form.get('text')

    image_result = None
    audio_result = None
    text_result = None

    if image_file:
        image_path = os.path.join('uploads', image_file.filename)
        image_file.save(image_path)
        image_result = process_image(image_path)

    if audio_file:
        audio_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_path)
        transcription, audio_output_path = audio_to_text_and_speech(audio_path)
        audio_result = {
            'transcription': transcription,
            'audio_output_path': audio_output_path
        }

    if text_input:
        text_result = process_text(text_input)

    return render_template('result.html', image_result=image_result, audio_result=audio_result, text_result=text_result)