#!/bin/bash
# Smart Webcam Assistant - camera-only GUI launcher
cd "$(dirname "$0")"
source venv/bin/activate
python gui_app.py
