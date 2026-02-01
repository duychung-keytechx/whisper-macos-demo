# Lightning Whisper STT Demo

Real-time Speech-to-Text demo for macOS Apple Silicon using Lightning-SimulWhisper.

## Features

- Real-time streaming speech-to-text
- Optimized for Apple Silicon (M1/M2/M3/M4)
- Uses MLX for efficient inference
- WebSocket-based streaming architecture
- React frontend with audio visualization

## Requirements

- macOS with Apple Silicon (M1, M2, M3, M4)
- Python 3.10+
- Node.js 18+

## Quick Start

### 1. Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install UI dependencies
cd ui && npm install && cd ..
```

### 2. Run the Backend

```bash
source venv/bin/activate
python api.py
```

The backend will start on `http://localhost:8000`.

### 3. Run the Frontend

In a new terminal:

```bash
cd ui
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Configuration

Environment variables:

- `WHISPER_MODEL`: Model size (tiny, base, small, medium, large-v3). Default: `base`
- `LANGUAGE`: Language code. Default: `en`
- `USE_COREML`: Enable CoreML acceleration (true/false). Default: `false`

Example:

```bash
WHISPER_MODEL=small LANGUAGE=en python api.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React)                        │
├─────────────────────────────────────────────────────────────┤
│  Microphone → Web Audio API → 16-bit PCM chunks             │
│                    ↓                                         │
│              WebSocket (ws://localhost:8000)                │
│                    ↓                                         │
│  Transcript updates ← Partial/Final messages                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                        │
├─────────────────────────────────────────────────────────────┤
│  WebSocket → Audio Buffer → SimulWhisper                    │
│                               ↓                              │
│                          MLX Inference                      │
│                               ↓                              │
│                    Real-time transcription                  │
└─────────────────────────────────────────────────────────────┘
```

## Models

The app uses MLX-optimized Whisper models from Hugging Face:

- `mlx-community/whisper-tiny-mlx` (~40MB)
- `mlx-community/whisper-base-mlx` (~150MB) - Default
- `mlx-community/whisper-small-mlx` (~500MB)
- `mlx-community/whisper-medium-mlx` (~1.5GB)
- `mlx-community/whisper-large-v3-mlx` (~3GB)

Models are automatically downloaded on first run.

## License

MIT
