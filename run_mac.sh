#!/bin/bash
set -e

echo "=========================================="
echo "  AirNox AI/ML Research MVP Setup (Mac)"
echo "=========================================="

python3 -m venv venv
source venv/bin/activate

echo "[1/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[2/4] Downloading & preparing free bioaerosol microscopy dataset..."
python3 -m src.data.download_dataset

echo "[3/4] Creating leak-free train/val/test splits..."
python3 -m src.data.preprocessing

echo "[4/4] Training ResNet18 baseline on Apple Silicon GPU (MPS)..."
python3 -m src.training.train

echo "=========================================="
echo "  Launching Interactive Streamlit App..."
echo "=========================================="
streamlit run app/streamlit_app.py
