from transformers import Blip2ForConditionalGeneration, AutoProcessor
import torch
import gradio as gr
from PIL import Image

# Load the model and processor
processor = AutoProcessor.from_pretrained("./BLIP_finetuned_dora_r16/checkpoint_26")
model = Blip2ForConditionalGeneration.from_pretrained("./BLIP_finetuned_dora_r16/checkpoint_26")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Function to generate caption for uploaded image
def generate_caption(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)  # Convert NumPy array to PIL Image if needed

    # Prepare image for model
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    # Generate caption
    generated_ids = model.generate(pixel_values=pixel_values, max_length=25)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Return the image and the generated caption
    return image, f"**Generated Caption:** {generated_caption}"

# Define Gradio interface with user-uploaded image input
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(label="Upload a Skin Image"),
    outputs=[gr.Image(label="Uploaded Image"), gr.Markdown(label="Generated Caption")],
    title="Skin Cancer Diagnosis",
    description="Upload an image to generate a skin cancer description below the image.",
)

# Launch the Gradio app
iface.launch(share=True)
