from transformers import Blip2ForConditionalGeneration, AutoProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# Load the model and processor from your saved directory
processor = AutoProcessor.from_pretrained("./BLIP_finetuned_dora_r16/checkpoint_26")
model = Blip2ForConditionalGeneration.from_pretrained("./BLIP_finetuned_dora_r16/checkpoint_26")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def generate_caption(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    # Generate caption
    generated_ids = model.generate(pixel_values=pixel_values, max_length=30)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Draw the caption on the image
    img = image.convert("RGB")  # Ensure RGB format
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Wrap text to fit within the image width
    max_width = img.width - 20  # Allow for padding
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

    # Calculate text height and position
    line_height = draw.textbbox((0, 0), "Test", font=font)[3] - draw.textbbox((0, 0), "Test", font=font)[1]
    text_height = line_height * len(lines)
    text_position = (10, img.height - text_height - 10)  # Bottom padding

    # Draw white background rectangle
    padding = 5
    rect_x1 = 5
    rect_y1 = text_position[1] - padding
    rect_x2 = img.width - 5
    rect_y2 = text_position[1] + text_height + padding
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill="white")

    # Draw each line of text in black
    for i, line in enumerate(lines):
        draw.text((10, text_position[1] + i * line_height), line, fill="black", font=font)

    return img


# Define the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(label="Image with Caption"),
    title="Skin Cancer Diagnosis",
    description="Upload an image to get a generated caption."
)

# Launch the Gradio app
iface.launch(share=True)
