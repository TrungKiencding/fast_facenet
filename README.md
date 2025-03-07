# Fast FaceNet API

A FastAPI-based face recognition system using FaceNet and MTCNN.

## Features

- Face Detection using MTCNN
- Face Recognition using FaceNet
- Multiple face angles support (front, left, right)
- SQLite database storage
- RESTful API endpoints

## Setup

1. Clone the repository
```bash
git clone https://github.com/TrungKiencding/fast_facenet
cd fast_facenet
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

## API Endpoints

- `GET /` - Welcome message
- `POST /recognize` - Recognize faces in image
- `POST /register` - Register new face
- `POST /check` - Check database entries
- `POST /delete` - Delete face entry

## Requirements

See `requirements.txt` for complete list of dependencies.

## Model Setup

1. Download the FaceNet model:
```bash
# Add instructions for downloading the model file
wget [model-url] -O fast_facenet.tflite
```

2. Place the downloaded `fast_facenet.tflite` file in the project root directory.
