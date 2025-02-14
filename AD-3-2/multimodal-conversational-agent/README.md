# Multimodal Conversational Agent

This project implements a multimodal conversational agent that processes images, audio, and text inputs in Telugu. The goal is to bridge the gap between traditional text-based systems and inclusive, context-aware solutions.

## Project Structure

```
multimodal-conversational-agent
├── app
│   ├── __init__.py          # Initializes the Flask application and sets up configurations and routes.
│   ├── routes.py            # Defines the routes for the web application.
│   ├── static
│   │   └── styles.css       # Contains CSS styles for the web application.
│   └── templates
│       ├── index.html       # Main HTML template for user input.
│       └── result.html      # Displays results of the multimodal processing.
├── models
│   ├── image_model.py       # Implements image processing model using pretrained Vision Transformers (ViTs).
│   ├── audio_model.py       # Implements audio processing model using Wav2Vec for speech-to-text conversion.
│   └── text_model.py        # Defines text processing model using IndicBERT or XLM-RoBERTa.
├── preprocessing
│   ├── image_preprocessing.py # Functions for image preprocessing.
│   ├── audio_preprocessing.py # Functions for audio preprocessing.
│   └── text_preprocessing.py  # Functions for text preprocessing.
├── fusion
│   └── fusion_model.py      # Implements the fusion layer for multimodal data.
├── utils
│   └── helpers.py           # Utility functions for various tasks.
├── main.py                  # Entry point of the application.
├── requirements.txt         # Lists dependencies required for the project.
└── README.md                # Documentation for the project.
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd multimodal-conversational-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

4. **Access the application:**
   Open your web browser and navigate to `http://127.0.0.1:5000`.

## Usage Guidelines

- Upload images, audio files, or enter text in Telugu to interact with the conversational agent.
- The agent will process the inputs and provide context-aware responses.

## Acknowledgments

This project utilizes various libraries and models, including:
- Flask for web framework
- PyTorch for deep learning
- Hugging Face Transformers for NLP tasks
- OpenCV and Librosa for preprocessing tasks

## License

This project is licensed under the MIT License. See the LICENSE file for more details.