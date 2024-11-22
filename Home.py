import random
import streamlit as st
from PIL import Image, ImageOps
# import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

st.title("Outpainting Images")
st.subheader("with the help of ControlNet")

controlnet = ControlNetModel.from_pretrained("destitech/controlnet-inpaint-dreamer-sdxl")

pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9",
    variant="fp16",
    controlnet=controlnet,
).to("cpu")

prompt = st.text_input("Kindly put in your prompt")

uploaded_image = st.file_uploader("Upload a base image (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

st.write("Expand to:")
left = st.checkbox("Left")
right = st.checkbox("Right")
up = st.checkbox("Up")
down = st.checkbox("Down")


if uploaded_image:
    seed = random.randint(0, 2**32 - 1)

    image = Image.open(uploaded_image).convert("RGB")
    border_left = 50 if left else 0
    border_up = 50 if up else 0
    border_right = 50 if right else 0
    border_down = 50 if down else 0

    padded_image = ImageOps.expand(image, border=(border_left, border_up, border_right, border_down), fill=(255, 255, 255))

    padded_image

