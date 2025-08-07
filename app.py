import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import numpy as np
import uuid
from PIL import Image, ImageOps
from model import DigitClassifier  # Import the model class

# Load model
@st.cache_resource
def load_model():
    model = DigitClassifier()
    model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Title and instructions
st.title("🧠 Digit Recognizer (MNIST)")
st.markdown("Draw a digit (0–9) below. Click **Submit Drawing** to predict it.")

# Ensure canvas_key is initialized
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_default"

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    submit_clicked = st.button("Submit Drawing")
with col2:
    reset_clicked = st.button("Reset")

# Reset button logic: assign new canvas key to clear
if reset_clicked:
    st.session_state.canvas_key = str(uuid.uuid4())

# Draw canvas
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

# Submit button logic
if submit_clicked:
    if canvas_result.image_data is not None:
        # Preprocess image
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = img.resize((28, 28))
        img = ImageOps.invert(img).convert('L')

        # Show processed image
        st.image(img, caption="Preprocessed Image (28x28)", width=100)

        # Prepare tensor
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()

        # Output
        st.subheader(f"Prediction: {pred}")
        st.write(f"Confidence: {confidence:.2%}")

        # Feedback form
        with st.form("feedback"):
            true_label = st.number_input("Correct digit (if different)", min_value=0, max_value=9, step=1)
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                st.success(f"Thanks! You entered: {true_label}. (This could be saved for retraining.)")
    else:
        st.warning("Please draw a digit before submitting.")
