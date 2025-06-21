import os
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTFeatureExtractor
from transformers import TrOCRProcessor

# -------- Configuration --------
MODEL_DIR = "final_model1"  # Path to trained model directory
IMAGE_DIR = "dataset/test"  # Path to folder with test images
OUTPUT_FILE = "predictions.txt"  # Output file to store results
TELUGU_TOKEN = "<2te>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load Model and Processor --------
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# Check token
if TELUGU_TOKEN not in tokenizer.get_vocab():
    raise ValueError(f"{TELUGU_TOKEN} token not found in tokenizer vocab.")

# -------- Inference --------
results = []
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

for image_name in image_files:
    image_path = os.path.join(IMAGE_DIR, image_name)

    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        pixel_values = processor.image_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        decoder_input_ids = torch.tensor([[tokenizer.convert_tokens_to_ids(TELUGU_TOKEN)]]).to(DEVICE)

        # Generate prediction
        output_ids = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        results.append(f"{image_name}\t{prediction}")
        print(f"{image_name} -> {prediction}")

    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")
        results.append(f"{image_name}\tERROR")

# -------- Save Predictions --------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print(f"\n✅ Predictions saved to {OUTPUT_FILE}")
