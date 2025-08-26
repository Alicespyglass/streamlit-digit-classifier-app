import os
import uuid
from datetime import datetime

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError

from model import DigitClassifier

from dotenv import load_dotenv

# Load variables from .env into os.environ
load_dotenv()

# --------------------
# Configuration
# --------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    st.error("DATABASE_URL environment variable not set. Please check your .env file.")
    st.stop()

# Initialize database engine in session state for a single, long-lived connection
if "db_engine" not in st.session_state:
    try:
        st.session_state.db_engine = create_engine(DATABASE_URL)
        
        # Define the SQL statement to create the table
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ,
                predicted_digit INTEGER,
                true_label INTEGER,
                image_data BYTEA,
                confidence REAL,
                is_correct BOOLEAN
            );
        """

        with st.session_state.db_engine.connect() as conn:
            # Use `text()` to execute a raw SQL command
            conn.execute(text(create_table_sql))
            conn.commit()  # Use conn.commit() to finalize the transaction
            
            # Use inspector to verify that the table was created
            inspector = inspect(conn)
            if 'predictions' in inspector.get_table_names():
                st.success("✅ Successfully connected to the database and ensured `predictions` table exists!")
            else:
                st.error("❌ Failed to create or find the `predictions` table.")
                st.stop()
    except OperationalError as e:
        st.error(f"❌ Failed to connect to the database. Check your DATABASE_URL in the .env file.")
        st.error(f"Error details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during database connection.")
        st.error(f"Error details: {e}")
        st.stop()

# --------------------
# Helper Functions
# --------------------
@st.cache_resource
def load_model():
    model = DigitClassifier()
    model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
    model.eval()
    return model

def preprocess_canvas_image(canvas_data):
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

def log_prediction(predicted, true_label=None, image_data=None, confidence=None):
    try:
        # Check if the user's feedback matches the model's prediction
        is_correct = (predicted == true_label)
        
        # Convert the numpy array to a bytes object for storage
        if image_data is not None:
            from io import BytesIO
            from PIL import Image
            img_buffer = BytesIO()
            img = Image.fromarray(image_data[:, :, 0]).convert('L')
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
        else:
            img_bytes = None

        with st.session_state.db_engine.begin() as conn:
            query = text(
                "INSERT INTO predictions (timestamp, predicted_digit, true_label, image_data, confidence, is_correct) "
                "VALUES (:ts, :pred, :true_label, :img_data, :conf, :is_correct)"
            )
            params = {
                "ts": datetime.now(), 
                "pred": predicted, 
                "true_label": true_label,
                "img_data": img_bytes,
                "conf": confidence,
                "is_correct": is_correct
            }
            
            conn.execute(query, params)
        st.session_state.feedback_success = True
    
    except Exception as e:
        st.error(f"❌ Failed to log prediction: {type(e).__name__} - {e}")
        st.warning("This could be a permissions issue, a typo in the table/column name, or a data type mismatch.")
        st.session_state.feedback_success = False

# --------------------
# Main Streamlit App
# --------------------
st.title("🧠 Digit Recognizer (MNIST)")
st.markdown("Draw a digit (0–9) below. Click **Submit Drawing** to predict it.")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_default"
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "feedback_success" not in st.session_state:
    st.session_state.feedback_success = False

col1, col2 = st.columns([1, 1])
with col1:
    submit_clicked = st.button("Submit Drawing", type="primary")
with col2:
    if st.button("Reset"):
        st.session_state.canvas_key = str(uuid.uuid4())
        st.session_state.prediction_made = False
        st.session_state.feedback_success = False
        st.rerun()

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

model = load_model()

# Check for empty canvas before making a prediction
if submit_clicked:
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        # User has drawn something, proceed with prediction
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
        st.session_state.feedback_success = False
        st.rerun()
    else:
        # Canvas is empty, show warning
        st.warning("Please draw a digit before submitting.")

# the feedback form
# In your main Streamlit App section, inside the feedback form
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
            # Pass the image data and confidence to the log function
            log_prediction(
                st.session_state.predicted_digit,
                true_label,
                canvas_result.image_data,
                st.session_state.confidence
            )
            if st.session_state.feedback_success:
                st.success("✅ Feedback logged successfully!")
            else:
                st.error("❌ Failed to log feedback. See errors above.")              
        
