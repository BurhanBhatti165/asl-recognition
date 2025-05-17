# ASL Letter Recognition Web Application

A real-time American Sign Language (ASL) letter recognition web application built with FastAPI and TensorFlow.

## Features

- Real-time ASL letter recognition using webcam
- Image upload for offline recognition
- Responsive design for all devices
- High-performance prediction with FPS counter
- Advanced image preprocessing for better accuracy

## Prerequisites

- Python 3.8 or higher
- Webcam (for real-time recognition)
- Modern web browser with WebRTC support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/asl-recognition.git
cd asl-recognition
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Use the application:
   - Click "Start Camera" to begin real-time recognition
   - Position your hand clearly in the camera view
   - Make ASL signs and see the predictions in real-time
   - Alternatively, upload an image using the file upload section

## Model Information

The application uses a TensorFlow model trained to recognize ASL letters. The model can identify:
- All 26 letters of the alphabet (A-Z)
- Additional signs (space, delete)

## Project Structure

```
asl-recognition/
├── app.py              # FastAPI application
├── static/
│   └── index.html      # Frontend interface
├── asl_model.h5        # Trained model
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

- FastAPI
- TensorFlow
- Pillow
- NumPy
- Uvicorn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the machine learning framework
- FastAPI team for the web framework
- All contributors and users of this project 