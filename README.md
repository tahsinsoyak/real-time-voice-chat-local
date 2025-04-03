# Real-Time Voice Chat with Transformers

## Overview
This project is a real-time voice chat application that integrates OpenAI's Whisper for speech recognition and a Turkish GPT-2-based model for generating responses. The system captures audio input, transcribes it, generates a response, and plays the response using text-to-speech. It is designed to act as a virtual tour guide for Istanbul, providing helpful and accurate information to users.

## Features
- **Speech Recognition**: Uses OpenAI's Whisper model for accurate transcription of Turkish speech.
- **Natural Language Processing**: Leverages a Turkish GPT-2-based model for generating meaningful and context-aware responses.
- **Text-to-Speech**: Converts generated responses into speech using Google Text-to-Speech (gTTS).
- **Real-Time Interaction**: Captures audio, processes it, and responds in real-time.
- **Fallback Responses**: Provides default responses when user input is unclear or the language model is unavailable.
- **GPU Support**: Automatically uses GPU if available for faster processing.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/tahsinsoyak/real-time-voice-chat-local.git
   ```
2. Navigate to the project directory:
   ```bash
   cd real-time-voice-chat-local
   ```
3. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```bash
      svenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
      source venv/bin/activate
     ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Speak into your microphone when prompted. The application will transcribe your speech, generate a response, and play it back.
3. To exit the program, say one of the following commands:
   - "çıkış"
   - "kapat"
   - "programı kapat"

## Dependencies
The project requires the following Python libraries:
- `numpy`
- `sounddevice`
- `pygame`
- `gTTS`
- `transformers`
- `torch`

For GPU usage, install PyTorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
```

For CPU-only usage:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Notes
- Ensure your system has the required audio drivers for microphone and speaker functionality.
- The application automatically selects GPU if available; otherwise, it defaults to CPU.
- The Whisper model and Turkish GPT-2 model are downloaded from Hugging Face's model hub during the first run.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
