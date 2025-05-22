

# Skin Cancer Detection Using BLIP Fine-Tuned with DoRA

This project implements a deep learning pipeline to detect and describe various skin cancers using **BLIP-2** (Bootstrapped Language-Image Pretraining) fine-tuned with **DoRA** (Weight-Decomposed Low-Rank Adaptation). It supports early diagnosis through image captioning and classification, with a Gradio-powered UI for user interaction.

---

## 📂 Dataset Preparation

We merge a skin cancer dataset with a general-purpose captioning dataset for improved robustness and generalization.

### 🔬 Skin Cancer Dataset

- **Source**: `Pranavkpba2000/skin_cancer_small_dataset`
- **Samples**: 14,200 across 8 classes (AK, BCC, BKL, MEL, NV, SCC, DF, VASC)
- Each label is mapped to a detailed medical-style caption

![WhatsApp Image 2025-03-11 at 14 52 42](https://github.com/user-attachments/assets/2a073769-db8a-435c-9961-19afab42648e)

*Figure: Example images from the skin cancer dataset showing various lesion types such as Actinic Keratosis, Basal Cell Carcinoma, Melanoma, and others, each labeled for supervised training.*


  


### 🖼️ Recap-COCO-30K Dataset

- **Source**: `UCSC-VLAA/Recap-COCO-30K`
- **Samples**: 6,000 randomly selected
- General-purpose captions for diversity and language fluency

### 🔗 Merge Code

```python
from datasets import load_dataset, concatenate_datasets

dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="train")
dataset = dataset.map(lambda x: {"caption": label_to_caption[x["label"]]}, num_proc=4)
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["image", "caption"]])

recap_coco = load_dataset("UCSC-VLAA/Recap-COCO-30K", split="train").shuffle(seed=42).select(range(6000))
recap_coco = recap_coco.remove_columns([col for col in recap_coco.column_names if col not in ["image", "caption"]])

merged_dataset = concatenate_datasets([dataset, recap_coco]).shuffle(seed=42)
````

---

## 🧠 Model Architecture

* **Model**: BLIP-2 with OPT-2.7B
* **Adaptation**: DoRA (rank=16, alpha=32)
* **Quantization**: 8-bit using `BitsAndBytes`
* **Frameworks**: PyTorch, HuggingFace, PEFT

---

## 🏋️ Training (`train.py`)

```bash
nohup python train.py > train.log &
```

* LoRA-based DoRA fine-tuning
* 50 Epochs, Batch Size: 3
* AdamW Optimizer with Cosine LR scheduler
* Automatic mixed-precision on CUDA (FP16)
* Early stopping and checkpoint saving

---

## 📊 Evaluation (`evaluate.py`)

Evaluate the model performance using BLEU, ROUGE, CIDEr, and METEOR:

```bash
python evaluate.py
```

* Evaluates on the skin cancer dataset's test split
* Automatically logs best scores to `best_scores.json`

---

## 💻 Gradio App (`live.py`)

Run the live interactive app for testing the model on new images:

```bash
python live.py
```

* Upload image
* Adjust `max_length`, `temperature`, `top_k`
  
<img width="1440" alt="Screenshot 2025-05-22 at 6 52 25 AM" src="https://github.com/user-attachments/assets/cef5cd60-57a2-421c-b44a-6ebcdd8b3491" />

*Figure: Gradio web interface for skin cancer detection. Users can upload an image of a skin lesion, adjust parameters like max length, temperature, and top-k, and receive an AI-generated medical-style caption describing the lesion.*



---

## ⚙️ Environment Setup

```bash
git clone https://github.com/yourusername/skin-cancer-blip-dora.git
cd skin-cancer-blip-dora

conda create -n skin-cancer-blip python=3.10 -y
conda activate skin-cancer-blip

pip install -r requirements.txt
```

---

## 🧪 Evaluation Metrics

| Metric  | Score  |
| ------- | ------ |
| BLEU    | 0.6062 |
| ROUGE-1 | 0.7189 |
| ROUGE-2 | 0.6368 |
| ROUGE-L | 0.7058 |
| CIDEr   | \~1.23 |
| METEOR  | \~0.52 |

---

## 🔮 Future Work

* Expand dataset coverage
* Integrate clinical metadata
* Enable on-device real-time inference
* Support multiple languages
* Incorporate explainability (Grad-CAM, SHAP)
* Connect with telemedicine APIs
* Continuous learning with feedback integration

---

## 📎 References

* [BLIP-2 (Salesforce)](https://github.com/salesforce/LAVIS)
* [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
* [PEFT - Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
* [DoRA: Weight-Decomposed Low-Rank Adaptation (2024)](https://arxiv.org/abs/2402.09353)


