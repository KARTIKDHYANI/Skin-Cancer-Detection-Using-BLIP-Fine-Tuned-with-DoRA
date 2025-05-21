from transformers import Blip2ForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import gradio as gr
from peft import PeftModel
from datasets import load_dataset
import torch



# Call this function when needed, e.g., before model loading or after inference.


# Load dataset
dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="test")

# Placeholder for model addresses (User will fill in the addresses)
MODEL_ADDRESSES = {
    "Version 1": "BLIP_finetuned_dora_r16_final_GOOD/best_checkpoint",
    "Version 2": "BLIP_finetuned_dora_r16_final_BETTER/best_checkpoint",
    "Version 3": "BLIP_finetuned_dora_r16_final_BESR/best_checkpoint",
    "Version 4": "BLIP_finetuned_dora_r16_final/best_checkpoint"
}

def load_model(model_path):
    base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = PeftModel.from_pretrained(base_model, model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor

# Default model load (User should update model path as needed)
selected_model_path = ""
model, processor = None, None

def update_model(version):
    global model, processor, selected_model_path
    selected_model_path = MODEL_ADDRESSES[version]
    model, processor = load_model(selected_model_path)
    return f"Model updated to: {version}"

def get_image_and_label(index):
    index=int(index)
    image = dataset[index]['image']
    label = dataset[index]['label']
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    return image, f"Label: {label}"

def generate_caption(image, max_length, temperature, top_k):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
    pixel_values = inputs.pixel_values
    
    generated_ids = model.generate(
        pixel_values=pixel_values, 
        max_new_tokens=max_length,
        temperature=temperature, 
        top_k=top_k,
        do_sample=True
    )
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return image, generated_caption

iface = gr.Interface(
    fn=get_image_and_label,
    inputs=gr.Number(label="Image Index"),
    outputs=[gr.Image(label="Image"), gr.Textbox(label="Label")],
    title="Skin Cancer Dataset Viewer",
    description="Enter an index to view an image with its corresponding label."
)

caption_iface = gr.Interface(
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
    description="Upload an image and adjust parameters to fine-tune the caption generation."
)

model_selector = gr.Interface(
    fn=update_model,
    inputs=gr.Radio(list(MODEL_ADDRESSES.keys()), label="Select Model Version"),
    outputs=gr.Textbox(label="Model Update Status"),
    title="Model Selector",
    description="Choose a model version to switch between different trained models."
)

gr.TabbedInterface([iface, caption_iface, model_selector], ["Dataset Viewer", "Caption Generator", "Model Selector"]).launch(share=True)