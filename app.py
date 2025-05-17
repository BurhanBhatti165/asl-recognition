from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging
import sys
from PIL import ImageEnhance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the application starts
model = None
model_path = 'asl_model.h5'

try:
    # Print Python and TensorFlow versions
    logger.info(f"Python version: {sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {os.path.abspath(model_path)}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Check file size
    file_size = os.path.getsize(model_path)
    logger.info(f"Model file size: {file_size / (1024*1024):.2f} MB")
    
    # Create a custom InputLayer class that handles batch_shape
    class CustomInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, input_shape=None, batch_shape=None, **kwargs):
            if batch_shape is not None:
                input_shape = batch_shape[1:]
            super().__init__(input_shape=input_shape, **kwargs)
    
    # Define custom objects to handle version compatibility
    custom_objects = {
        'InputLayer': CustomInputLayer,
        'DTypePolicy': tf.keras.mixed_precision.Policy
    }
    
    # Try to load the model with custom_objects
    logger.info(f"Attempting to load model from: {os.path.abspath(model_path)}")
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
    
    # Compile the model with appropriate settings
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model loaded successfully")
    
    # Save the working model with a new name
    working_model_path = 'asl_model_working.h5'
    model.save(working_model_path)
    logger.info(f"Working model saved to: {os.path.abspath(working_model_path)}")
    
    # Print model summary
    model.summary(print_fn=logger.info)
    
except FileNotFoundError as e:
    logger.error(f"File not found error: {str(e)}")
    raise Exception(f"Model file not found: {str(e)}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(f"Error type: {type(e).__name__}")
    raise Exception(f"Failed to load model: {str(e)}")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

def preprocess_image(image_data):
    # Convert to PIL Image
    img = Image.open(io.BytesIO(image_data))
    
    # Resize to model's expected input size
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    
    # Convert to RGB
    img = img.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # Increase contrast by 50%
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Apply samplewise centering and standardization
    img_array = img_array - np.mean(img_array, axis=(0, 1), keepdims=True)
    img_array = img_array / (np.std(img_array, axis=(0, 1), keepdims=True) + 1e-6)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs for details.")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(contents)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)  # Disable prediction logging
        predicted_class = np.argmax(prediction[0])
        
        # Define the classes
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                  'del', 'nothing', 'space']
        
        # Get the predicted letter and confidence
        predicted_letter = classes[predicted_class]
        confidence = float(prediction[0][predicted_class])
        
        # Only return predictions with confidence above threshold
        if confidence < 0.5:  # 50% confidence threshold
            return JSONResponse({
                "prediction": "No clear sign detected",
                "confidence": confidence
            })
        
        return JSONResponse({
            "prediction": predicted_letter,
            "confidence": confidence
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

