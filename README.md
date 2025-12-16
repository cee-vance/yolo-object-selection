

# 🎯 YOLO Object Selection

## Overview
YOLO Object Selection is a **computer vision project** that uses YOLOv5 for real‑time object detection and interactive selection. Users can select detected objects via console input, laser pointer, or voice commands. The project demonstrates applied machine learning, multimodal interfaces, and integration of pretrained models.

## Features
- **YOLOv5 object detection** with pretrained weights  
- **Multiple selection modes**: console, laser pointer, voice input  
- **Real‑time inference** on video streams or images  
- **Modular design** for extending detection classes and interfaces  

## Tech Stack
- **Frameworks:** PyTorch, OpenCV  
- **Model:** YOLOv5 pretrained weights  
- **Languages:** Python  
- **Interfaces:** Console, voice recognition (SpeechRecognition), laser pointer input  

## Getting Started
```bash
# Clone the repo
git clone https://github.com/cee-vance/yolo-object-selection.git
cd yolo-object-selection

# Install dependencies
pip install -r requirements.txt

# Run detection
python detect.py --source 0  # webcam
