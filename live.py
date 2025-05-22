from transformers import Blip2ForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import gradio as gr
from peft import PeftModel

# Step 1: Load the base BLIP-2 model
base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Step 2: Load the adapter weights (fine-tuned checkpoint)
model = PeftModel.from_pretrained(base_model, "BLIP_finetuned_dora_r16_final/best_checkpoint")

# IMPORTANT: Use the processor from the base model, not the adapter
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Caption generation function
def generate_caption(image, max_length, temperature, top_k):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    prompt = "a medical image of"  # or just "" for generic

    # Preprocess image and prompt
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    # Generate caption
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True
    )
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return image, generated_caption

# Gradio Interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Slider(10, 100, value=30, step=5, label="Max Length"),
        gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(0, 100, value=50, step=5, label="Top-k Sampling")
    ],
    outputs=[
        gr.Image(label="Uploaded Image"),
        gr.Textbox(label="Generated Caption")
    ],
    title="Skin Cancer Diagnosis",
    description="Upload a skin lesion image and adjust parameters to generate a medical caption."
)

# Launch the app
iface.launch(share=True)
