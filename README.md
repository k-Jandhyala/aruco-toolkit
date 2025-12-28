# ArUco Marker Detection

A Python application for detecting ArUco markers in images or from a webcam feed using OpenCV. Available as both a command-line tool and a web application.

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or if you're using a virtual environment:

```bash
source venv/bin/activate  # On macOS/Linux
pip3 install -r requirements.txt
```

## Web Application (FastAPI)

### Running the Web Server

Start the FastAPI web application:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser and navigate to:

```
http://localhost:8000
```

### Web Features

- **Modern Web Interface**: Beautiful, responsive UI for uploading images or using webcam
- **Webcam Support**: Capture frames directly from your webcam and detect markers
- **Drag & Drop Support**: Easily upload images by dragging and dropping
- **Real-time Detection**: Instant ArUco marker detection and visualization
- **Marker ID Display**: See all detected marker IDs prominently displayed
- **Dictionary Selection**: Choose from multiple ArUco dictionary types
- **Result Visualization**: View the processed image with detected markers highlighted

### Using Webcam

1. Click on the "📷 Use Webcam" tab
2. Click "🎥 Start Webcam" to begin (browser will ask for camera permission)
3. Position ArUco markers in front of the camera
4. Click "📸 Capture & Detect" to capture the current frame and detect markers
5. View the detected marker IDs and processed image
6. Click "⏹️ Stop Webcam" when done

## Command-Line Usage

### Detect from Webcam

```bash
python main.py --camera
```

- Press `q` to quit
- Press `s` to save the current frame

### Detect from Image File

```bash
python main.py --image path/to/your/image.jpg
```

The result will be saved with `_aruco` added to the filename.

### Specify Dictionary Type

You can specify different ArUco dictionary types:

```bash
python main.py --image image.jpg --dictionary DICT_4X4_50
```

Available dictionary types:
- `DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000`
- `DICT_5X5_50`, `DICT_5X5_100`, `DICT_5X5_250`, `DICT_5X5_1000`
- `DICT_6X6_50`, `DICT_6X6_100`, `DICT_6X6_250`, `DICT_6X6_1000` (default)
- `DICT_7X7_50`, `DICT_7X7_100`, `DICT_7X7_250`, `DICT_7X7_1000`

## Features

- **Web Application**: FastAPI-based web interface with modern UI
- **Real-time Detection**: ArUco marker detection from webcam or images
- **Image Processing**: Upload and process image files
- **Multiple Dictionary Support**: Choose from various ArUco dictionary types
- **Visual Marker ID Display**: See detected marker IDs clearly displayed
- **Automatic Result Saving**: Results are automatically processed and displayed

## Requirements

- opencv-python >= 4.8.0
- opencv-contrib-python >= 4.8.0
- numpy >= 1.24.0
- fastapi >= 0.104.0
- uvicorn[standard] >= 0.24.0
- python-multipart >= 0.0.6
- aiofiles >= 23.2.0
- jinja2 >= 3.1.2

