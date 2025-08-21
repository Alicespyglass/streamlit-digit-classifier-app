import os
import uuid
from datetime import datetime

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from sqlalchemy import create_engine, text

from model import DigitClassifier

from dotenv import load_dotenv

# Load variables from .env into os.environ
load_dotenv()

# --------------------
# Configuration
# --------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    st.error("DATABASE_URL environment variable not set")
    st.stop()

# Initialize database engine in session state for a single, long-lived connection
if "db_engine" not in st.session_state:
    st.session_state.db_engine = create_engine(DATABASE_URL)

# --------------------
# Helper Functions
# --------------------
@st.cache_resource
def load_model():
    """Load the pre-trained PyTorch model."""
    model = DigitClassifier()
    model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
    model.eval()
    return model

def preprocess_canvas_image(canvas_data):
    """Preprocess the canvas data into a 28x28 grayscale tensor."""
    img = Image.fromarray(canvas_data[:, :, 0].astype('uint8')).convert('L')
    img = ImageOps.invert(img)
    img = img.point(lambda x: 0 if x < 50 else 255, mode='L')
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    new_img = Image.new('L', (28, 28), 0)
    upper_left = ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2)
    new_img.paste(img, upper_left)
    img_tensor = torch.tensor(np.array(new_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return new_img, img_tensor

def log_prediction(predicted, true_label=None):
    """Log prediction to the database."""
    try:
        with st.session_state.db_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO predictions (timestamp, predicted_digit, true_label) "
                    "VALUES (:ts, :pred, :true_label)"
                ),
                {"ts": datetime.now(), "pred": predicted, "true_label": true_label}
            )
        st.info(f"✅ Logged prediction={predicted}, true_label={true_label} to DB")
        with st.session_state.db_engine.connect() as conn:
            last = conn.execute(text("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")).fetchone()
            st.write("Last row in DB:", last)
    except Exception as e:
        st.error(f"❌ Failed to log prediction: {e}")

# --------------------
# Main Streamlit App
# --------------------
st.title("🧠 Digit Recognizer (MNIST)")
st.markdown("Draw a digit (0–9) below. Click **Submit Drawing** to predict it.")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_default"
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    submit_clicked = st.button("Submit Drawing", type="primary")
with col2:
    if st.button("Reset"):
        st.session_state.canvas_key = str(uuid.uuid4())
        st.session_state.prediction_made = False
        st.rerun()

# Canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key
)

# Load model
model = load_model()

# Prediction logic
if submit_clicked and canvas_result.image_data is not None:
    img, img_tensor = preprocess_canvas_image(canvas_result.image_data)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    st.session_state.predicted_digit = pred
    st.session_state.confidence = confidence
    st.session_state.processed_image = img
    st.session_state.prediction_made = True
    st.rerun()

# Display prediction and feedback form if a prediction has been made
if st.session_state.prediction_made:
    st.subheader(f"Prediction: {st.session_state.predicted_digit}")
    st.write(f"Confidence: {st.session_state.confidence:.2%}")
    st.image(st.session_state.processed_image, caption="Processed 28x28 Image", width=100)

    with st.form(key="feedback_form"):
        st.write("👇 Provide feedback here:")
        true_label = st.number_input(
            "Correct digit (if different)",
            min_value=0, max_value=9, step=1,
            value=st.session_state.predicted_digit
        )
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            log_prediction(st.session_state.predicted_digit, true_label)
            st.success(f"Thanks for your feedback! It helps improve the model.")
            st.session_state.prediction_made = False
            st.rerun()

else:
    if submit_clicked:
        st.warning("Please draw a digit before submitting.")