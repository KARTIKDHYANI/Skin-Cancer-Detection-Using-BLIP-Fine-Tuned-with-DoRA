from transformers import Blip2ForConditionalGeneration, AutoProcessor
import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# Load the model and processor
processor = AutoProcessor.from_pretrained("./BLIP_finetuned_dora_r16_final")
model = Blip2ForConditionalGeneration.from_pretrained("./BLIP_finetuned_dora_r16_final")
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

    # Draw the caption on the image
    img = image.convert("RGB")  # Ensure RGB format
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Wrap text for better readability
    max_width = img.width - 20
    words = generated_caption.split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}" if line else word
        test_width = draw.textbbox((0, 0), test_line, font=font)[2] - draw.textbbox((0, 0), test_line, font=font)[0]
        if test_width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    # Position text at the bottom
    line_height = draw.textbbox((0, 0), "Test", font=font)[3] - draw.textbbox((0, 0), "Test", font=font)[1]
    text_height = line_height * len(lines)
    text_position = (10, img.height - text_height - 10)

    for i, line in enumerate(lines):
        draw.text((10, text_position[1] + i * line_height), line, fill="white", font=font)

    return img

# Define Gradio interface with user-uploaded image input
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(label="Upload a Skin Image"),
    outputs=gr.Image(label="Image with Caption"),
    title="Skin Cancer Diagnosis",
    description="Upload an image to generate a skin cancer description.",
)

# Launch the Gradio app
iface.launch(share=True)
