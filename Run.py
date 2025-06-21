import os
import torch
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, TrOCRProcessor
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from evaluate import load as load_metric
import albumentations as A
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Set Telugu font for Matplotlib
telugu_font_path = "/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf"  # Adjust path if needed
if os.path.exists(telugu_font_path):
    prop = fm.FontProperties(fname=telugu_font_path)
else:
    print("Telugu font not found. Install Lohit Telugu or use another Telugu font.")
    prop = fm.FontProperties()

# File paths
train_text_file = "C:/Users/petas/Desktop/OCR/dataset/train.txt"
test_text_file = "C:/Users/petas/Desktop/OCR/dataset/test.txt"
val_text_file = "C:/Users/petas/Desktop/OCR/dataset/val.txt"
root_dir = "C:/Users/petas/Desktop/OCR/dataset/"
checkpoint_dir = "final_model1"

# Normalize Telugu text
normalizer = IndicNormalizerFactory().get_normalizer("te")
def normalize_df(df):
    df['text'] = df['text'].apply(lambda x: normalizer.normalize(x))
    return df

# Read dataset
def dataset_generator(data_path):
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    dataset = []
    for line in lines:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            dataset.append([parts[0], parts[1]])
    return pd.DataFrame(dataset, columns=["file_name", "text"])

train_df = normalize_df(dataset_generator(train_text_file))
test_df = normalize_df(dataset_generator(test_text_file))
val_df = normalize_df(dataset_generator(val_text_file))

print(f"Train, Test & Val shape: {train_df.shape, test_df.shape, val_df.shape}")

# Verify dataset (visualize 5 samples)
for i in range(5):
    img_path = os.path.join(root_dir, train_df.iloc[i]['file_name'])
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(train_df.iloc[i]['text'], fontproperties=prop)
    plt.axis('off')
    plt.savefig(f"sample_{i}.png")
    plt.close()

# Dataset class
class OCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.GaussNoise(p=0.3),
            A.Resize(224, 224)  # Match ViT model
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.df.loc[idx, 'file_name'])
        text = "<2te> " + self.df.loc[idx, 'text'] + " </s>"
        image = Image.open(file_path).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = self.transform(image=np.array(image))['image']
        image = Image.fromarray(image)
        pixel_values = self.processor.image_processor(images=image, return_tensors="pt").pixel_values
        encoding = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length,
                                            truncation=True, return_tensors="pt")
        labels = encoding.input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values.squeeze(0),
            "labels": labels.clone().detach()  # Safer tensor copy
        }

# Model names
encoder_model = 'google/vit-base-patch16-224-in21k'
decoder_model = 'ai4bharat/IndicBARTSS'

# Processor
feature_extractor = ViTImageProcessor.from_pretrained(encoder_model, size=224)
tokenizer = AutoTokenizer.from_pretrained(decoder_model, use_fast=True, do_lower_case=False, keep_accents=True)
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

# Check tokenizer coverage
unique_chars = set("".join(train_df['text']))
missing_chars = [c for c in unique_chars if c not in tokenizer.get_vocab()]
print("Missing characters:", missing_chars)
if missing_chars:
    tokenizer.add_tokens(list(missing_chars))

# Datasets
train_dataset = OCRDataset(root_dir, train_df, processor)
val_dataset = OCRDataset(root_dir, val_df, processor)
test_dataset = OCRDataset(root_dir, test_df, processor)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize or load model and processor
training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
start_epoch = 0
best_val_cer = float('inf')
train_losses = []
val_cers = []

if os.path.exists(training_state_path):
    # Load processor
    processor = TrOCRProcessor.from_pretrained(checkpoint_dir)
    # Load model
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    # Load training state
    checkpoint = torch.load(training_state_path, map_location=device)
    start_epoch = checkpoint["epoch"]
    best_val_cer = checkpoint["best_val_cer"]
    train_losses = checkpoint.get("train_losses", [])
    val_cers = checkpoint.get("val_cers", [])
    print(f"Resuming training from epoch {start_epoch}, Best Val CER: {best_val_cer:.4f}")
    print(f"Loaded train_losses: {train_losses}")
    print(f"Loaded val_cers: {val_cers}")
else:
    # Initialize new model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)
    if missing_chars:
        model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("<2te>")
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Starting fresh training")

model.to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_loader) * 30
)

# Load optimizer state if resuming
if os.path.exists(training_state_path):
    checkpoint = torch.load(training_state_path, map_location=device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Mixed precision
scaler = torch.amp.GradScaler('cuda')

# Metrics
cer_metric = load_metric("cer")
wer_metric = load_metric("wer")

def evaluate(model, loader, epoch):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="Validating"):
        with torch.no_grad():
            pixel_values = batch['pixel_values'].to(device)
            labels_batch = batch['labels'].to(device)
            generated_ids = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=torch.full((pixel_values.shape[0], 1), tokenizer.convert_tokens_to_ids("<2te>"), dtype=torch.long).to(device),
                max_length=128, num_beams=4, early_stopping=True
            )
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels_batch[labels_batch == -100] = tokenizer.pad_token_id
            decoded_labels = tokenizer.batch_decode(labels_batch, skip_special_tokens=True)
            if len(preds) == 0:
                print(f"Epoch {epoch+1} Sample Pred: {decoded_preds[0]}")
                print(f"Epoch {epoch+1} Sample Label: {decoded_labels[0]}")
            preds.extend(decoded_preds)
            labels.extend(decoded_labels)
    cer = cer_metric.compute(predictions=preds, references=labels)
    wer = wer_metric.compute(predictions=preds, references=labels)
    print(f"CER: {cer:.4f}, WER: {wer:.4f}")
    return cer

# Training
num_epochs = 30
accumulation_steps = 4

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        with torch.amp.autocast('cuda'):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")

    val_cer = evaluate(model, val_loader, epoch)
    val_cers.append(val_cer)

    if val_cer < best_val_cer:
        best_val_cer = val_cer
        print("Saving best model")
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        torch.save({
            "epoch": epoch + 1,
            "best_val_cer": best_val_cer,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_cers": val_cers
        }, training_state_path)

    # Visualize CER and Loss
    # Adjust plotting to use the actual length of train_losses and val_cers
    num_epochs_so_far = len(train_losses)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs_so_far + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs_so_far + 1), val_cers, label="Validation CER")
    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.title("Validation CER")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"epoch_{epoch+1}_metrics.png")
    plt.close()

# Final test CER
test_cer = evaluate(model, test_loader, epoch)
print(f"Final Test CER: {test_cer:.4f}")
