import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import numpy as np
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

# Canvas for user input
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Submit button
if st.button("Submit Drawing"):
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

        # Feedback
        with st.form("feedback"):
            true_label = st.number_input("Correct digit (if different)", min_value=0, max_value=9, step=1)
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                st.success(f"Thanks! You entered: {true_label}. (This could be saved for retraining.)")
    else:
        st.warning("Please draw a digit before submitting.")
