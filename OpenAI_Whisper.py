#!pip install --upgrade pip
#!pip install --upgrade transformers datasets[audio] evaluate jiwer accelerate soundfile librosa

#!pip install "datasets<4.0.0" --force-reinstall
#!pip install --upgrade transformers evaluate jiwer accelerate soundfile librosa

import os
import pandas as pd
from datasets import Dataset, Audio
from google.colab import drive

drive.mount('/content/drive')

tar_path = "/content/drive/MyDrive/Colab Notebooks/Anonymous-3/mcv-scripted-tr-v23.0.tar.gz"
extract_path = "/content/data"

if not os.path.exists(extract_path):
    print(f"Extracting file to: {extract_path}...")
    #!mkdir -p "{extract_path}"
    #!tar -xzf "{tar_path}" -C "{extract_path}"
    print("Extraction complete!")
else:
    print("Directory already exists, skipping extraction.")

try:
    extracted_folders = [f.path for f in os.scandir(extract_path) if f.is_dir()]
    if not extracted_folders:
        raise FileNotFoundError("Archive extracted but appears empty.")

    base_folder = extracted_folders[0]

    if "tr" not in os.listdir(base_folder):
         sub_folders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
         if sub_folders:
             base_folder = sub_folders[0]

    lang_folder = os.path.join(base_folder, "tr")

    if not os.path.exists(lang_folder):
         print("Warning: 'tr' folder not found in standard path.")

    tsv_path = os.path.join(lang_folder, "validated.tsv")
    clips_folder = os.path.join(lang_folder, "clips")

    print(f"TSV File found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Successfully loaded! Total rows: {len(df)}")

except Exception as e:
    print("Error occurred:", e)

from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os

# --- Data Preparation ---
# Re-creating dataset variables to ensure fresh start
df_clean = df[["path", "sentence"]].copy()
df_clean["audio"] = df_clean["path"].apply(lambda x: os.path.join(clips_folder, x))
df_clean = df_clean.drop(columns=["path"])

# Demo için 2000 veri
df_clean = df_clean.head(2000)

dataset = Dataset.from_pandas(df_clean)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.train_test_split(test_size=0.1)

print("Dataset prepared!")

# --- Processor ---
model_id = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_id, language="Turkish", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("Mapping dataset features...")
dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

# --- Metrics ---
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- Model & Training ---
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False # Training sırasında cache kapalı olmalı

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tr-final",
    per_device_train_batch_size=4,  # Reduced to 4 to prevent OOM
    gradient_accumulation_steps=4,  # Increased to 4 to compensate
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=400,
    gradient_checkpointing=False,   # DISABLED: This fixes the RuntimeError
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor, # Updated for new version warning
)

print("Starting training...")
trainer.train()