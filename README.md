# Real-Time Voice Chat with Transformers

## Overview
A real-time voice chat application that integrates OpenAI's Whisper for speech recognition and a Turkish GPT-2 based model for generating responses. The system captures audio input, transcribes it, generates a response, and plays the response using text-to-speech.
This project leverages Hugging Face's Transformers and model hub to load and run state-of-the-art models.

## Installation
1. Clone the repository.
   ```
   git clone https://github.com/tahsinsoyak/real-time-voice-chat-local.git
   ```
3. Create a Python virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the application with:
```
python app.py
```
Speak into your microphone when prompted. To exit the program, say "çıkış", "kapat", or "programı kapat".

## Dependencies
- numpy
- sounddevice
- pygame
- gTTS
- transformers
- torch

## Notes
- Ensure your system has the required audio drivers.
- The application selects GPU if available, otherwise it defaults to CPU.
