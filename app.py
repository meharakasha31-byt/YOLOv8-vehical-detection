import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
import os

# Define the path to the trained model weights relative to the app.py location
model_path = 'best.pt'

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Vehicle Detection App",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for styling (placeholder for user's existing UI theme)
# This uses Streamlit's markdown to inject CSS, attempting to match the requested theme.
st.markdown(
    '''
<style>
.stApp {
    background-color: #2e2e2e; /* Dark background, similar to purple/black */
    color: #f0f0f0; /* Light text color */
}
h1, h2, h3, h4, h5, h6 {
    color: #a766ff; /* Purple-like color for headings */
}
.stButton>button {
    background-color: #a766ff;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
    cursor: pointer;
}
.stButton>button:hover {
    background-color: #8c4ad9;
    color: white;
}
.stFileUploader label {
    color: #f0f0f0;
}
/* Further customization can go here to match existing UI */
</style>
''',
    unsafe_allow_html=True
)

st.title("🚗 Vehicle Detection with YOLOv8")
st.write("Upload an image to detect vehicles using a pre-trained YOLOv8 model.")

@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_model():
    try:
        # Load the model directly from the 'best.pt' file in the same directory
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    img_np = np.array(image) # Convert PIL image to NumPy array

    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Detecting objects...")

    if model:
        # Perform inference
        results = model(img_np, conf=0.25, verbose=False) # verbose=False to suppress print output

        # Process and display results
        for r in results:
            # Draw bounding boxes and labels on the image
            im_array = r.plot() # plot a BGR numpy array of predictions
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB) # Convert to RGB for display
            st.image(im_rgb, caption='Detected Objects', use_column_width=True)

            # Optionally display detection details
            if r.boxes:
                st.subheader("Detection Details:")
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = model.names[cls] # Get class name from model
                    st.write(f"- **Class:** {label}, **Confidence:** {conf:.2f}, **BBox:** ({x1}, {y1}, {x2}, {y2})")
            else:
                st.write("No objects detected.")
    else:
        st.error("Model could not be loaded. Please check the setup.")

st.caption("Powered by YOLOv8 and Streamlit")
