#!/bin/bash

# Setup script for Lightning Whisper STT Demo

set -e

echo "Setting up Lightning Whisper STT Demo..."

# Create and activate virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Pre-download the Whisper model (optional, will be downloaded on first run)
echo "Pre-downloading Whisper model (base)..."
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading mlx-community/whisper-base-mlx...')
snapshot_download('mlx-community/whisper-base-mlx')
print('Model downloaded successfully!')
"

# Install UI dependencies
echo "Installing UI dependencies..."
cd ui
npm install
cd ..

echo ""
echo "Setup complete!"
echo ""
echo "To run the app:"
echo "  1. Start the backend: source venv/bin/activate && python api.py"
echo "  2. Start the frontend: cd ui && npm run dev"
echo ""
echo "The app will be available at http://localhost:5173"
