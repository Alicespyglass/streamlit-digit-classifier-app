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
        # Step 1: Convert canvas to grayscale image and invert colors
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype('uint8')).convert('L')
        img = ImageOps.invert(img)

        # Step 2: Binarize the image
        img = img.point(lambda x: 0 if x < 50 else 255, mode='L')

        # Step 3: Crop to bounding box if available
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Step 4: Resize to fit 20x20 while keeping aspect ratio
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)

        # Step 5: Pad to 28x28
        new_img = Image.new('L', (28, 28), 0)  # black background
        upper_left = ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2)
        new_img.paste(img, upper_left)

        # Step 6: Convert to normalized tensor [1, 1, 28, 28]
        img_tensor = torch.tensor(np.array(new_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        # Debug: Show processed image
        # st.image(new_img, caption="Processed 28x28 Image", width=100)
        # st.write("Input tensor shape:", img_tensor.shape)
        # st.write("Min pixel value:", img_tensor.min().item())
        # st.write("Max pixel value:", img_tensor.max().item())

        # Step 7: Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()

        # Debug - Display probabilities
        # for i, p in enumerate(probs[0]):
        #     st.write(f"{i}: {p:.2%}")

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
