from datasets import load_dataset, concatenate_datasets

# Load Skin Cancer dataset
dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="train")

# Define label-to-caption mapping
label_to_caption = {
    0: "The detected disease is Actinic Keratosis (AK), a precancerous condition characterized by scaly, crusty patches of skin. It can potentially develop into skin cancer if not treated.",
    1: "The detected disease is Basal Cell Carcinoma (BCC), a common form of skin cancer that typically appears as a pearly or waxy bump. It is slow-growing and usually doesn't spread.",
    2: "The detected disease is Benign Keratosis (BKL), a non-cancerous skin growth that often appears as a wart or mole. These are usually harmless but may need monitoring.",
    3: "The detected disease is Melanoma (MEL), a serious and aggressive form of skin cancer that often presents as a mole with irregular edges or different colors. It requires immediate attention.",
    4: "The detected disease is Nevus (NV), commonly known as a mole, which can vary in color and size. Most are harmless, but changes in size, shape, or color may require evaluation.",
    5: "The detected disease is Squamous Cell Carcinoma (SCC), a type of skin cancer that often appears as a firm, red nodule or scaly patch. It can spread if not treated early.",
    6: "The detected disease is Dermatofibroma (DF), a benign growth on the skin that is typically brown or tan. These are harmless and generally do not require treatment.",
    7: "The detected disease is Vascular Lesions (VASC), abnormal blood vessel growths that can appear as red or purple spots on the skin. They are generally harmless but may require treatment for cosmetic reasons."
}

# Add caption column based on label
dataset = dataset.map(lambda example: {"caption": label_to_caption[example["label"]]}, num_proc=4)

# Load Recap-COCO-30K dataset and sample 5000 rows
recap_coco_dataset = load_dataset("UCSC-VLAA/Recap-COCO-30K", split="train").shuffle(seed=42).select(range(6000))   

# Keep only 'image' and 'caption' columns in both datasets
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["image", "caption"]])
recap_coco_dataset = recap_coco_dataset.remove_columns([col for col in recap_coco_dataset.column_names if col not in ["image", "caption"]])

# Merge both datasets
merged_dataset = concatenate_datasets([dataset, recap_coco_dataset]).shuffle(seed=42)  # Final shuffle

# Print dataset details
print(merged_dataset)
print(merged_dataset[0])  # Print a sample row  


from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["caption"] = item["caption"]
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "caption":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["caption"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", quantization_config=quant_config)

from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"],
    use_dora=True
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


train_dataset = ImageCaptioningDataset(merged_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=3, collate_fn=collate_fn)

import torch
from torch.optim import AdamW
from transformers import get_scheduler

# Training parameters
e = 50
output_dir = "./BLIP_finetuned_dora_r16_final"
gradient_accumulation_steps = 4
best_loss = float("inf")
patience = 5
no_improve_epochs = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
# Define optimizer

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Compute training steps
num_training_steps = len(train_dataloader) * e
num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup

# Learning rate scheduler
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# Training loop
for epoch in range(e):
    print(f"Epoch {epoch + 1}/{e}")
    model.train()
    total_loss = 0
    
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss / gradient_accumulation_steps

        loss.backward()
        total_loss += loss.item()

        if (idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if idx % 50 == 0:
            print(f"  Step {idx}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve_epochs = 0
        model.save_pretrained(f"{output_dir}/best_checkpoint")
        processor.save_pretrained(f"{output_dir}/best_checkpoint")
        print(f"✅ Best model saved at epoch {epoch+1}")
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        print(f"❌ Early stopping triggered at epoch {epoch+1}")
        break

# Save final model
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"🎉 Final model saved at {output_dir}")
