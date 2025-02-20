import pickle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

# FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://work-advisor.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model
model = load_model('models/engagement_predictor_model.keras')

# Load saved tokenizer, scaler, and category encoder
with open("artifacts/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("artifacts/title_tokenizer.pkl", "rb") as f:
    title_tokenizer = pickle.load(f)

with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("artifacts/category_encoder.pkl", "rb") as f:
    category_encoder = pickle.load(f)

# Define the class mapping
class_mapping = {
    0: 'Decent',
    1: 'Popular',
    2: 'Unpopular'
}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    
    post_content = data.get("post_content")
    post_title = data.get("post_title")
    category = data.get("category")
    
    if not post_content or not post_title or not category:
        return JSONResponse(content={"error": "Missing input data!"}, status_code=400)

    # Tokenize and pad post content
    post_sequence = tokenizer.texts_to_sequences([post_content])
    post_padded = pad_sequences(post_sequence, maxlen=234)

    # Tokenize and pad post title
    title_sequence = title_tokenizer.texts_to_sequences([post_title])
    title_padded = pad_sequences(title_sequence, maxlen=13)

    # One-hot encode the category
    category_encoded = category_encoder.transform(np.array([[category]]))

    predictions = []
    # Predict for different days: 1, 3, and 7 days
    for days_since_post in [1, 3, 7]:
        # Scale the "days_since_post" value
        days_scaled = scaler.transform(np.array([[days_since_post]]))

        # Make prediction
        prediction = model.predict([post_padded, title_padded, days_scaled, category_encoded])

        # Get the predicted class (inverse transform)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        
        # Map the predicted class index to the actual class label
        predicted_class_label = class_mapping.get(predicted_class_index, "Unknown")

        predictions.append({
            "days_since_post": days_since_post,
            "predicted_class": predicted_class_label
        })

    return JSONResponse(content={"predictions": predictions})
