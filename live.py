from transformers import Blip2ForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import gradio as gr
from peft import PeftModel

# Step 1: Load the base BLIP-2 model
base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Step 2: Load the adapter weights
model = PeftModel.from_pretrained(base_model,"BLIP_finetuned_dora_r16_final/best_checkpoint")
# Load the model and processor
processor = AutoProcessor.from_pretrained("BLIP_finetuned_dora_r16_final/best_checkpoint")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def generate_caption(image, max_length, temperature, top_k):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    # Generate caption with adjustable parameters
    generated_ids = model.generate(
        pixel_values=pixel_values, 
        max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
        temperature=temperature, 
        top_k=top_k,
        do_sample=True
    )
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return image, generated_caption  # Return image and caption separately

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Slider(10, 100, value=30, step=5, label="Max Length"),
        gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(0, 100, value=50, step=5, label="Top-k Sampling")
    ],
    outputs=[
        gr.Image(label="Uploaded Image"),  # Display original image
        gr.Textbox(label="Generated Caption")  # Display caption separately
    ],
    title="Skin Cancer Diagnosis",
    description="Upload an image and adjust parameters to fine-tune the caption generation."
)

# Launch the Gradio app
iface.launch(share=True)
