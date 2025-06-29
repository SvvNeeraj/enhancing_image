import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from RRDN_arch import RRDN

# Page setup
st.set_page_config(page_title="RRDN Super-Resolution", layout="centered")
st.title(" RRDN Image Super-Resolution ")

# Load RRDN model
@st.cache_resource
def load_model():
    model = RRDN(in_nc=3, out_nc=3, nf=64, nb=16, gc=32, scale=4)
    model.load_state_dict(torch.load("rrdn_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image preprocessing/postprocessing
def preprocess_image(image):
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor)

# File upload
uploaded_file = st.file_uploader("Upload a low-resolution image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Input Image", use_container_width=True)

    enhance = st.button(" Enhance ")

    if enhance:
        with st.spinner("Enhancing image..."):
            input_tensor = preprocess_image(input_image)
            with torch.no_grad():
                output_tensor = model(input_tensor)
            output_image = postprocess_image(output_tensor)

        # Save enhanced image
        output_path = "enhanced_output.png"
        output_image.save(output_path)

        # Prepare images for comparison
        original_resized = input_image.resize(output_image.size)
        original_np = np.array(original_resized).astype(np.float32) / 255.0
        enhanced_np = np.array(output_image).astype(np.float32) / 255.0

        # Compute metrics
        psnr_val = peak_signal_noise_ratio(original_np, enhanced_np, data_range=1.0)
        ssim_val = structural_similarity(original_np, enhanced_np, channel_axis=-1, data_range=1.0)

        # Quality metrics
        st.subheader(" Image Quality Metrics ")
        st.markdown(f"**PSNR:** {psnr_val:.2f} dB")
        st.markdown(f"**SSIM:** {ssim_val:.4f}")

        # Side-by-side image display
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_resized, caption="Original (Resized)",use_container_width=True)
        with col2:
            st.image(output_image, caption="Enhanced", use_container_width=True)

        # Download button
        with open(output_path, "rb") as f:
            st.download_button(
                label=" Download Enhanced Image ",
                data=f,
                file_name="enhanced_output.png",
                mime="image/png"
            )
