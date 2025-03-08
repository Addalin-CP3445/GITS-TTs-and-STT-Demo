import gradio as gr
import whisper
import torch
import pyaudio
import wave
import neuralspace as ns
import requests
from queue import Queue
from TTS.api import TTS
import shutil
import base64

from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])

# Initialize Whisper and Coqui models
whisper_model = whisper.load_model("base")
device = "cuda" if torch.cuda.is_available() else "cpu"
coqui_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# API keys
VOICEAI_API_KEY = "sk_aa55ba99d40a1e3256b0b8ce78de35cd0a8a07e7d1d9071bdffb2b7923f6195d"
ELEVENLABS_API_KEY = "sk_c45e05b9c306fd53df5b5ec1d987331ebdf0ceb91a5e0d9d"
AIVOOV_API_KEY = "7bd5aa8e-8487-471a-aca1-0247883e336e"

# Voice AI setup
vai = ns.VoiceAI(api_key=VOICEAI_API_KEY)
q = Queue()


# Whisper STT
def transcribe_whisper(audio_path):
    audio = whisper.load_audio("./" + audio_path)
    result = whisper_model.transcribe(audio)
    return result["text"]

# Voice AI STT
def transcribe_voiceai(audio_path):
    config = {
    'file_transcription': {
        'mode': 'advanced',
        },
    }

    try:
        job_id = vai.transcribe(file=audio_path, config=config)
        result = vai.poll_until_complete(job_id)
        return result["data"]["result"]["transcription"]["channels"]["0"]["transcript"]
    
    except Exception as e:
        return ""

# ElevenLabs STT
def transcribe_elevenlabs(audio_path):
    url = "https://api.elevenlabs.io/v1/speech-to-text"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }

    files = {
    "file": open(audio_path, "rb")  
    }

    data = {
    "model_id": "scribe_v1",
    "language_code": "ar"
    }

    response = requests.post(url, headers=headers, files=files, data=data)

    text_data = ""
    if response.ok:
        text_data = response.json()
        return text_data["text"]
    else:
        return text_data






# Coqui TTS
def generate_tts_coqui(text):
    output_file = "coqui_output_audio.wav"
    try:
        coqui_tts.tts_to_file(text=text, file_path=output_file, speaker_wav="V2.wav", language="ar")
        return output_file
    except Exception as e:
        return "silent.wav"

# aivoov TTS
def generate_tts_aivoov(text):
    filepath = ""
    url = "https://aivoov.com/api/v1/transcribe"

    headers = {
        "X-API-KEY": AIVOOV_API_KEY,
    }

    payload = {
        "voice_id": "ar-SA-HamedNeural",
        "transcribe_text[]": text,  
        "engine": "neural"
    }

    response = requests.post(url, data=payload, headers=headers)
    response_json = response.json()

    try:
        base64_audio = response_json["transcribe_data"]

        audio_bytes = base64.b64decode(base64_audio)

        filepath = "aivoov_output_audio.mp3"
        with open(filepath, "wb") as audio_file:
            audio_file.write(audio_bytes)
    
        return filepath
    except Exception as e:
        return "silent.wav"

# ElevenLabs TTS
def generate_tts_elevenlabs(text):
    voice_id="a1KZUXKFVFDOb33I1uqr"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_flash_v2_5",
                "voice_settings": {
                    "stability": 0.4,
                    "similarity_boost": 0.77
                },
        "language_code":"ar"
    }
    
    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    response = requests.post(url, json=data, headers=headers)
    output_file_path = ''
    try:
        output_file_path = 'elevenlabs_output_audio.mp3'
        with open(output_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return output_file_path
    except Exception as e:
        output_file_path = 'silent.wav'
        return output_file_path



# Gradio UI
audio_input = gr.Audio(type="filepath", label="Upload Audio for STT")
text_input = gr.Textbox(label="Enter text for TTS")
output_text = gr.Textbox(label="Transcription Result")
audio_output = gr.Audio(label="Generated Speech")

# Function to switch between STT models
def transcribe(audio_path, model):
    if audio_path is not None:
        file_path = "gradio_saved_audio.wav"
        
        shutil.copy(audio_path, file_path)

        if model == "Whisper":
            return transcribe_whisper(file_path)
        elif model == "Voice AI":
            return transcribe_voiceai(file_path)
        elif model == "Elevenlabs":
            return transcribe_elevenlabs(file_path)
        return "Invalid Model Selection"
    else:
        return "Invalid audio file"

# Function to switch between TTS models
def synthesize(text, model):
    if text is not None or bool(text.strip()):
        if model == "Coqui TTS":
            return generate_tts_coqui(text)
        elif model == "AiVOOV":
            return generate_tts_aivoov(text)
        elif model == "Elevenlabs":
            return generate_tts_elevenlabs(text)
        return "silent.wav"
    else:
        return "silent.wav"

demo = gr.Interface(
    title="Speech-to-Text and Text-to-Speech Demo",
    description="Choose STT and TTS models to transcribe or generate speech.",
    inputs=[
        gr.Radio(["Whisper", "Voice AI", "Elevenlabs"], label="Select STT Model"),
        audio_input,
        gr.Radio(["Coqui TTS", "AiVOOV", "Elevenlabs"], label="Select TTS Model"),
        text_input
    ],
    outputs=[output_text, audio_output],
    fn=lambda stt_model, audio, tts_model, text: (
        transcribe(audio, stt_model),
        synthesize(text, tts_model)
    ),
)

demo.launch()
