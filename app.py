import os
# Ortam ayarları
os.environ["HF_HUB_TIMEOUT"] = "120"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import queue
import tempfile
import numpy as np
import torch
import sounddevice as sd
import pygame
from gtts import gTTS
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline,
    WhisperProcessor, WhisperForConditionalGeneration
)
# Initialize pygame for audio playback
pygame.mixer.init()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model for speech recognition
whisper_model_name = "openai/whisper-small"
try:
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    sys.exit(1)

# Turkish text generation model
model_name = "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    language_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    generator = pipeline(
        'text-generation',
        model=language_model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    model_loaded = True
except Exception as e:
    print(f"Error loading language model: {e}")
    model_loaded = False

# Fallback responses
fallback_responses = {
    "greeting": "Merhaba! Ben İstanbul sanal tur rehberinizim. Size nasıl yardımcı olabilirim?",
    "not_understood": "Üzgünüm, sizi anlayamadım. Lütfen tekrar eder misiniz?",
    "default": "İstanbul'da gezilecek birçok tarihi ve turistik yer var. Size özel bir yer hakkında bilgi vermemi ister misiniz?"
}

# Audio recording queue
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def record_audio(duration=10, samplerate=16000):
    """Record audio for specified duration and return as numpy array."""
    audio_data = []
    with sd.RawInputStream(samplerate=samplerate, blocksize=4000, dtype='int16',
                           channels=1, callback=audio_callback):
        start_time = time.time()
        while time.time() - start_time < duration:
            if not audio_queue.empty():
                audio_data.append(audio_queue.get())
    audio_np = np.frombuffer(b''.join(audio_data), dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np, samplerate

def transcribe_audio(audio_data, samplerate=16000):
    """Convert recorded audio to text using Whisper model."""
    if len(audio_data) == 0 or np.max(np.abs(audio_data)) < 0.01:
        return None
    try:
        input_features = whisper_processor(
            audio_data,
            sampling_rate=samplerate,
            return_tensors="pt"
        ).input_features.to(device)
        predicted_ids = whisper_model.generate(
            input_features,
            language="tr",
            task="transcribe"
        )
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def generate_response(user_text):
    """Generate a meaningful response to user text using language model."""
    if not user_text:
        return fallback_responses["not_understood"]
    if not model_loaded:
        return fallback_responses["default"]
    
    prompt = (
        "Sen İstanbul'da profesyonel bir tur rehberisin. Turistlere İstanbul hakkında doğru ve yararlı bilgiler veriyorsun.\n\n"
        f"Turist: {user_text}\n"
        "Rehber:"
    )
    
    try:
        response = generator(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )[0]['generated_text']
        
        if "Rehber:" in response:
            response_text = response.split("Rehber:")[-1].strip()
        else:
            response_text = response[len(prompt):].strip()
            
        if "Turist:" in response_text:
            response_text = response_text.split("Turist:")[0].strip()
            
        if len(response_text.split()) < 3:
            return fallback_responses["default"]
            
        return response_text
    except Exception as e:
        print(f"Response generation error: {e}")
        return fallback_responses["default"]

def speak_text(text):
    """Convert text to speech and play it."""
    if not text:
        return
    try:
        tts = gTTS(text=text, lang='tr', slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            audio_file = fp.name
        tts.save(audio_file)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        os.remove(audio_file)
    except Exception as e:
        print(f"Text-to-speech error: {e}")

def main():
    # Initial greeting
    speak_text("Merhaba! Ben İstanbul tur rehberinizim. Size nasıl yardımcı olabilirim?")
    
    while True:
        print("\nListening mode: Please speak for 10 seconds...")
        audio_data, sr = record_audio(duration=10)
        user_text = transcribe_audio(audio_data, sr)
        
        if user_text:
            print(f"User: {user_text}")
            if user_text.lower().strip() in ["çıkış", "kapat", "programı kapat"]:
                speak_text("Görüşmek üzere! İyi günler.")
                break
            response_text = generate_response(user_text)
            print(f"\nGuide: {response_text}")
            speak_text(response_text)
        else:
            print("No speech detected or too short.")
            
    print("Program terminated.")

if __name__ == "__main__":
    main()