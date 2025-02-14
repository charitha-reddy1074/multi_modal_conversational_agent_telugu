from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import binascii
import logging
from app.routes import app_routes
from utils.helpers import text_to_speech, convert_audio_to_text

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Generate a secret key
secret_key = binascii.hexlify(os.urandom(24)).decode()

def create_app():
    app = Flask(__name__, template_folder='app/templates')
    app.config['SECRET_KEY'] = secret_key
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.register_blueprint(app_routes)

    return app

app = create_app()
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    logging.debug("Received audio chunk")
    try:
        # Save the audio chunk temporarily
        file_path = os.path.join('temp', 'live_audio.wav')
        with open(file_path, 'wb') as f:
            f.write(data)
        logging.debug(f"Audio chunk saved to {file_path}")

        # Convert voice input to text
        text_output = convert_audio_to_text(file_path)
        logging.debug(f"Converted text: {text_output}")

        # Generate voice output from text
        audio_output_path = text_to_speech(text_output)
        logging.debug(f"Generated audio output path: {audio_output_path}")

        emit('audio_response', {'text': text_output, 'audio': audio_output_path})
    except Exception as e:
        logging.error(f"Error processing audio chunk: {e}")

if __name__ == "__main__":
    socketio.run(app, debug=True)