# api.py - FastAPI WebSocket server for Lightning-SimulWhisper
import asyncio
import json
import sys
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Add Lightning-SimulWhisper to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Lightning-SimulWhisper'))

from simul_whisper.config import AlignAttConfig
from simulstreaming_whisper import SimulWhisperOnline
import mlx.core as mx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")  # tiny, base, small, medium, large-v3
MODEL_PATH = os.getenv("MODEL_PATH", f"mlx_{MODEL_NAME}")
LANGUAGE = os.getenv("LANGUAGE", "en")
USE_COREML = os.getenv("USE_COREML", "false").lower() == "true"

# Audio processing constants
SAMPLE_RATE = 16000
CHUNK_DURATION_SECONDS = 0.3  # Process audio every N seconds (lower is more responsive but need to be considered)

# Global model
whisper_model = None

# Fine-grained lock for model inference (protects concurrent access)
model_inference_lock = asyncio.Lock()


def load_model():
    """Load the SimulWhisper model"""
    global whisper_model

    logger.info(f"Loading SimulWhisper model: {MODEL_NAME}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Language: {LANGUAGE}")
    logger.info(f"CoreML: {USE_COREML}")

    cfg = AlignAttConfig(
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        segment_length=CHUNK_DURATION_SECONDS,
        frame_threshold=25,
        language=LANGUAGE,
        audio_max_len=30.0,
        audio_min_len=0.5,
        cif_ckpt_path="",
        decoder_type="greedy",
        beam_size=1,
        task="transcribe",
        never_fire=True,  # Don't truncate last word
        init_prompt=None,
        max_context_tokens=None,
        static_init_prompt=None,
        logdir="logdir",
        vad_silence_ms=500,
        use_coreml_encoder=USE_COREML,
        coreml_encoder_path=None,
        coreml_compute_units="ALL",
    )

    from simul_whisper.simul_whisper import PaddedAlignAttWhisper
    whisper_model = PaddedAlignAttWhisper(cfg)

    logger.info("Model loaded successfully!")


class ClientSession:
    """Per-client session using SimulWhisperOnline"""

    SAMPLING_RATE = 16000
    MAX_AUDIO_SECONDS = 25  # Reset model before 30s buffer fills

    def __init__(self, model):
        # Create a wrapper that mimics the ASR interface
        class ASRWrapper:
            def __init__(self, m):
                self.model = m

        self.model = model
        self.online = SimulWhisperOnline(ASRWrapper(model))
        self.pending_samples = 0
        self.total_audio_seconds = 0
        self.last_text = ""

    def add_audio(self, audio_chunk):
        """Add audio chunk. Returns flushed text if reset occurred, None otherwise."""
        chunk_duration = len(audio_chunk) / self.SAMPLING_RATE
        reset_text = None

        # Check if we need to reset BEFORE adding this chunk to avoid losing audio
        if self.total_audio_seconds + chunk_duration >= self.MAX_AUDIO_SECONDS:
            logger.info(f"Resetting model after {self.total_audio_seconds:.1f}s of audio")

            # Flush any pending transcription before reset
            try:
                beg, end, text = self.online.finish()
                if text and text.strip():
                    reset_text = text.strip()
                    logger.info(f"Flushed before reset: '{reset_text}'")
            except Exception as e:
                logger.warning(f"Error flushing before reset: {e}")

            self.online.init()  # Reset the online processor
            self.total_audio_seconds = 0
            self.pending_samples = 0
            self.last_text = ""
            mx.clear_cache()

        # Now add the audio chunk (to fresh buffer if reset occurred)
        self.online.insert_audio_chunk(audio_chunk)
        self.pending_samples += len(audio_chunk)
        self.total_audio_seconds += chunk_duration

        # Clear cache frequently to prevent memory buildup
        mx.clear_cache()

        return reset_text

    def should_process(self):
        """Check if we have enough audio to process"""
        return self.pending_samples >= self.SAMPLING_RATE * CHUNK_DURATION_SECONDS

    def process(self):
        """Process and get transcription - returns raw model output"""
        self.pending_samples = 0
        beg, end, text = self.online.process_iter()

        # Clear MLX cache to prevent memory buildup
        mx.clear_cache()

        if text and text.strip():
            new_text = text.strip()
            if new_text != self.last_text:
                self.last_text = new_text
                logger.info(f"Model output: '{new_text}'")
                return new_text
        return None

    def finish(self):
        """Finalize transcription"""
        beg, end, text = self.online.finish()
        if text and text.strip():
            return text.strip()
        return None

    def reset(self):
        """Reset for new session"""
        self.online.init()
        self.pending_samples = 0
        self.total_audio_seconds = 0
        self.last_text = ""
        mx.clear_cache()


# Per-client sessions
client_sessions = {}


@app.on_event("startup")
async def startup_event():
    load_model()


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)

    # Create session for this client
    session = ClientSession(whisper_model)
    client_sessions[client_id] = session

    try:
        while True:
            message = await websocket.receive()

            # Handle text messages (ping/pong)
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                except json.JSONDecodeError:
                    continue

            # Handle binary audio data
            if "bytes" in message:
                data = message["bytes"]
                # Convert bytes to numpy array (assuming 16-bit PCM)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # Fine-grained lock: only held during model inference operations
                async with model_inference_lock:
                    # add_audio returns flushed text if a reset occurred
                    reset_text = session.add_audio(audio_chunk)

                    # Process audio when enough samples accumulated
                    text = None
                    if session.should_process():
                        text = session.process()

                # Send results outside the lock (I/O doesn't need lock)
                if reset_text:
                    await websocket.send_json({
                        "type": "chunk",
                        "text": reset_text
                    })

                if text:
                    # Send raw model output as "chunk" - frontend will accumulate
                    await websocket.send_json({
                        "type": "chunk",
                        "text": text
                    })

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Get final transcription and send it
        if client_id in client_sessions:
            try:
                # Acquire lock for final inference
                async with model_inference_lock:
                    final_text = session.finish()
                if final_text:
                    try:
                        await websocket.send_json({
                            "type": "final",
                            "text": final_text
                        })
                    except:
                        pass
            except Exception as e:
                logger.error(f"Error getting final transcription: {e}")

            del client_sessions[client_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
