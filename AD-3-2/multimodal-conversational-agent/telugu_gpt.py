# Install required libraries
#!pip install -q transformers torch sounddevice numpy scipy sentence-transformers gtts streamlit pydub pillow

#to run the file : streamlit run telugu_gpt.py

import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from gtts import gTTS
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from PIL import Image
import numpy as np
import io
import warnings
import streamlit as st

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(page_title="తెలుగు మల్టీమోడల్ చాట్‌బాట్", layout="wide")

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
@st.cache_resource
def load_models():
    print("Loading models...")
    try:
        # Embedding model for semantic search
        embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
        
        # Speech recognition model
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=0 if torch.cuda.is_available() else -1)
        
        # Text generation model
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)
        
        # Image captioning model
        img_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        img_captioner = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        
        print("Models loaded successfully!")
        return embedder, asr, tokenizer, model, img_processor, img_captioner
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

embedder, asr, tokenizer, model, img_processor, img_captioner = load_models()

# Knowledge Base
# Knowledge Base
knowledge_base = {
    # Temples
    "తిరుపతి": {
        "description": ("తిరుపతిలోని శ్రీ వేంకటేశ్వర స్వామి ఆలయం ప్రసిద్ధ హిందూ దేవాలయం. "
                      "ఇది ఆంధ్రప్రదేశ్ లోని తిరుమల పర్వతాల మీద ఉంది. "
                      "ప్రతి సంవత్సరం లక్షలాది భక్తులు ఇక్కడకు దర్శనం కోసం వస్తారు."),
        "tags": ["తిరుమల", "వేంకటేశ్వర", "దేవాలయం"]
    },
    "వారణాసి": {
        "description": ("వారణాసి (కాశీ) గంగా నది ఒడ్డున ఉన్న పుణ్యనగరం. "
                      "ఇది ప్రపంచంలోనే అత్యంత పురాతన నగరాలలో ఒకటి. "
                      "కాశీ విశ్వనాథ్ దేవస్థానం ఇక్కడి ప్రధాన ఆకర్షణ."),
        "tags": ["కాశీ", "గంగా", "విశ్వనాథ్"]
    },
    
    # Historical Sites
    "తాజ్ మహల్": {
        "description": ("తాజ్ మహల్ ఉత్తరప్రదేశ్ లోని ఆగ్రాలో ఉన్న పాలరాయి మక్బరా. "
                      "మొఘల్ చక్రవర్తి షాజహాన్ తన భార్య ముమ్తాజ్ మహల్ కోసం నిర్మించారు. "
                      "ఇది ప్రపంచ ఏడు అద్భుతాలలో ఒకటిగా పరిగణించబడుతుంది."),
        "tags": ["ఆగ్రా", "మొఘల్", "ప్రేమ స్మారకం"]
    },
    "ఢిల్లీ": {
        "description": ("భారతదేశ రాజధాని ఢిల్లీ చారిత్రక ప్రాముఖ్యత కలిగిన నగరం. "
                      "ఇక్కడ ఎర్రకోట, కుతుబ్ మినార్, లోటస్ టెంపుల్ వంటి ప్రసిద్ధ స్మారకాలు ఉన్నాయి. "
                      "ఇది దేశం యొక్క రాజకీయ, సాంస్కృతిక కేంద్రం."),
        "tags": ["రాజధాని", "ఎర్రకోట", "కుతుబ్ మినార్"]
    },
    
    # Natural Wonders
    "కేరళ": {
        "description": ("కేరళను 'దేవుని స్వంత నాడు' అని పిలుస్తారు. "
                      "బ్యాక్ వాటర్స్, కొక్కోనట్ బీచ్‌లు, అడవులు, ఆయుర్వేదం ఇక్కడి ప్రత్యేకతలు. "
                      "మున్నార్, అల్లెప్పీ, కొచ్చి ప్రముఖ పర్యాటక స్థలాలు."),
        "tags": ["బ్యాక్ వాటర్స్", "మున్నార్", "ఆయుర్వేదం"]
    },
    "లడఖ్": {
        "description": ("లడఖ్ హిమాలయాలలో ఉన్న ఒక శీతల మరుభూమి. "
                      "పంగాంగ్ సరోవర్, నుబ్రా వ్యాలీ, బుద్ధ మఠాలు ఇక్కడి ప్రధాన ఆకర్షణలు. "
                      "ఇది ట్రెక్కింగ్ మరియు సాహస యాత్రలకు ప్రసిద్ధి చెందింది."),
        "tags": ["హిమాలయాలు", "లేహ్", "ట్రెక్కింగ్"]
    },
    
    # Cities
    "ముంబై": {
        "description": ("ముంబై భారతదేశ ఆర్థిక రాజధాని. "
                      "గేట్వే ఆఫ్ ఇండియా, మరీన్ డ్రైవ్, బాలీవుడ్ ఇక్కడి ప్రధాన ఆకర్షణలు. "
                      "ఇది భారతదేశంలోని అత్యధిక జనసాంద్రత కలిగిన నగరం."),
        "tags": ["బాలీవుడ్", "గేట్వే", "ఆర్థిక రాజధాని"]
    },
    "బెంగళూరు": {
        "description": ("బెంగళూరును 'సిలికాన్ వ్యాలీ ఆఫ్ ఇండియా' అని పిలుస్తారు. "
                      "ఇది భారతదేశ ఐటీ రాజధాని. "
                      "లాల్ బాగ్, బెంగళూరు ప్యాలెస్, ఐఎస్‌కెఆన్ ఇక్కడి ప్రధాన ఆకర్షణలు."),
        "tags": ["ఐటీ", "సిలికాన్ వ్యాలీ", "గార్డెన్ సిటీ"]
    },
    
    # Pilgrimage
    "అమృతసర్": {
        "description": ("అమృతసర్ లోని స్వర్ణదేవాలయం (గోల్డెన్ టెంపుల్) సిక్కు మతానికి ప్రధాన పవిత్ర స్థలం. "
                      "ఇక్కడి లంగర్ (సామూహిక భోజనం) ప్రసిద్ధి చెందింది. "
                      "ఇది పంజాబ్ రాష్ట్రంలో ఉంది."),
        "tags": ["గోల్డెన్ టెంపుల్", "సిక్కులు", "లంగర్"]
    },
    "రామేశ్వరం": {
        "description": ("రామేశ్వరం తమిళనాడులో ఉన్న పవిత్ర హిందూ తీర్థయాత్రా కేంద్రం. "
                      "రామనాథస్వామి దేవస్థానం ఇక్కడి ప్రధాన ఆకర్షణ. "
                      "హిందూ పురాణాల ప్రకారం శ్రీరాముడు ఇక్కడే రావణుని వద్దకు సేతువు నిర్మించాడు."),
        "tags": ["సేతు", "రామనాథస్వామి", "తీర్థయాత్ర"]
    },
    
    # Added more places
    "హైదరాబాద్": {
        "description": ("హైదరాబాద్ తెలంగాణ రాజధాని. "
                      "చార్మినార్, గోల్కొండ కోట, రామోజీ ఫిల్మ్ సిటీ ఇక్కడి ప్రధాన ఆకర్షణలు. "
                      "ఇది భారతదేశ ఫార్మా మరియు ఐటీ హబ్."),
        "tags": ["చార్మినార్", "బిర్యానీ", "టెక్ సిటీ"]
    },
    "జైపూర్": {
        "description": ("జైపూర్ను 'పింక్ సిటీ' అని పిలుస్తారు. "
                      "హవా మహల్, అంబర్ కోట, జంతర్ మంతర్ ఇక్కడి ప్రధాన ఆకర్షణలు. "
                      "ఇది రాజస్థాన్ రాజధాని మరియు గోల్డెన్ ట్రయాంగిల్ లో భాగం."),
        "tags": ["పింక్ సిటీ", "హవా మహల్", "రాజస్థాన్"]
    }
}

# Pre-compute embeddings for the knowledge base
for item in knowledge_base.values():
    item["embedding"] = embedder.encode(item["description"])

def retrieve_knowledge(query, top_k=2):
    """Retrieve relevant knowledge using semantic search."""
    # Check for exact matches in the knowledge base
    for key, item in knowledge_base.items():
        if key in query or any(tag in query for tag in item["tags"]):
            return [item["description"]]

    # If no exact match, use semantic similarity
    query_embedding = embedder.encode(query)
    similarities = []
    
    for item in knowledge_base.values():
        sim = cosine_similarity([query_embedding], [item["embedding"]])[0][0]
        similarities.append((sim, item["description"]))
    
    # Sort by similarity and filter by a stricter threshold
    similarities.sort(reverse=True, key=lambda x: x[0])
    results = [text for (sim, text) in similarities[:top_k] if sim > 0.5]  # Increased threshold to 0.5

    # Debugging: Log the retrieved results
    if not results:
        print(f"No relevant knowledge found for query: {query}")
    return results

def generate_response(user_input):
    """Generate context-aware response."""
    context = retrieve_knowledge(user_input)
    
    if context:
        return " ".join(context)
    else:
        return "క్షమించండి, సమాధానం ఇవ్వలేకపోయాను."

def text_to_speech(text, lang='te'):
    """Convert text to speech using Google TTS."""
    if not text or text.strip() == "":
        text = "క్షమించండి, సమాధానం ఇవ్వలేకపోయాను."  # Default fallback text
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

def generate_image_caption(image):
    """Generate caption for an input image."""
    try:
        inputs = img_processor(image, return_tensors="pt").to(device)
        output = img_captioner.generate(**inputs)
        caption = img_processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Image captioning error: {e}")
        return "క్షమించండి, చిత్రానికి వివరణ ఇవ్వలేకపోయాను."

# Streamlit UI
st.title("తెలుగు మల్టీమోడల్ చాట్‌బాట్")
st.markdown("మీరు టెక్స్ట్, ఆడియో లేదా చిత్రాలను ఉపయోగించి చాట్ చేయండి!")

# Tabs for different input modes
tab1, tab2, tab3 = st.tabs(["టెక్స్ట్ ఇన్పుట్", "ఆడియో ఇన్పుట్", "చిత్ర ఇన్పుట్"])

# Text Input Tab
with tab1:
    st.subheader("టెక్స్ట్ ఇన్పుట్")
    user_input = st.text_input("మీ సందేశాన్ని ఇక్కడ టైప్ చేయండి:")
    if user_input:
        response = generate_response(user_input)
        st.write(f"*సమాధానం:* {response}")
        audio_response = text_to_speech(response)
        st.audio(audio_response, format="audio/mp3")

# Audio Input Tab
with tab2:
    st.subheader("ఆడియో ఇన్పుట్")
    audio_file = st.file_uploader("ఆడియో ఫైల్‌ను అప్‌లోడ్ చేయండి (wav/mp3):", type=["wav", "mp3"])
    if audio_file:
        audio = AudioSegment.from_file(audio_file)
        audio_data = np.array(audio.get_array_of_samples())
        transcript = asr(audio_data)["text"]
        st.write(f"*ట్రాన్స్‌క్రిప్షన్:* {transcript}")
        response = generate_response(transcript)
        st.write(f"*సమాధానం:* {response}")
        audio_response = text_to_speech(response)
        st.audio(audio_response, format="audio/mp3")

# Image Input Tab
with tab3:
    st.subheader("చిత్ర ఇన్పుట్")
    image_file = st.file_uploader("చిత్రాన్ని అప్‌లోడ్ చేయండి (jpg/png/jpeg):", type=["jpg", "png", "jpeg"])
    if image_file:
        image = Image.open(image_file).convert("RGB")
        caption = generate_image_caption(image)
        st.write(f"*చిత్ర వివరణ:* {caption}")
        response = generate_response(caption)
        st.write(f"*సమాధానం:* {response}")
        audio_response = text_to_speech(response)
        st.audio(audio_response, format="audio/mp3")