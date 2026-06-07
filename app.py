import io
import requests
import streamlit as st
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="centered")

# 2. Configuration for Hugging Face API
# Replace with your preferred model endpoint if desired, e.g.,
# "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
API_URL = (
    "https://api.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
)


from huggingface_hub import InferenceClient

def query_hf_api(prompt, api_key):
    try:
        # Initializes the official client
        client = InferenceClient(provider="hf-inference", token=api_key)
        
        # Requests the image generation directly
        image = client.text_to_image(
            prompt, 
            model="black-forest-labs/FLUX.1-schnell"
        )
        
        # The library safely returns a PIL Image object directly!
        return image
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# 3. Streamlit UI Elements
st.title("🎨 AI Text-to-Image Generator")
st.write(
    "Turn your imagination into art using free Hugging Face models and Streamlit."
)

# Sidebar for credentials and settings
with st.sidebar:
    st.header("Settings")
    hf_token = st.text_input(
        "Enter Hugging Face Token:",
        type="password",
        help="Get your token from huggingface.co/settings/tokens",
    )
    st.caption(
        "Note: Your token is processed locally and never stored permanently."
    )

# Main input text area
prompt = st.text_area(
    "Describe the image you want to generate:",
    placeholder="A futuristic city in the middle of a desert at sunset, cyberpunk style, highly detailed...",
)

# Generation Trigger
# Updated snippet for your button logic if using huggingface_hub:
if st.button("Generate Image", type="primary"):
    # ... (keep token validation steps the same) ...
    with st.spinner("Generating your image..."):
        img_result = query_hf_api(prompt, hf_token)

        if img_result:
            st.success("Generated successfully!")
            st.image(img_result, caption=f'Result for: "{prompt}"', use_container_width=True)
            
            # For the download button:
            buffered = io.BytesIO()
            img_result.save(buffered, format="PNG")
            st.download_button(
                label="Download Image",
                data=buffered.getvalue(),
                file_name="ai_generated_image.png",
                mime="image/png"
            )