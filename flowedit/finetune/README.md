# XTTS Fine-Tuning Pipeline for FlowEdit

This directory contains a complete fine-tuning pipeline to teach XTTS-v2 perfect pronunciation for custom Indian names, proper nouns, and foreign words.

---

## Complete Workflow

### Step 1: Prepare Raw Audio & Metadata
Place your audio recordings and metadata file inside a directory (e.g. `./raw_dataset`).

Create a file named `metadata.csv` inside `./raw_dataset`:
```text
wavs/sakharwade_1.wav|My name is Mrunmayee Sakharwade.|speaker1
wavs/sakharwade_2.wav|Hello, I am Mrunmayee Sakharwade.|speaker1
wavs/sakharwade_3.wav|Dr. Sakharwade is visiting us today.|speaker1
wavs/general_1.wav|The weather today is pleasant.|speaker1
```

### Step 2: Format Dataset
Run `prepare_dataset.py` to resample all audio to 22050 Hz Mono WAVs and split into `tts_train.csv` and `tts_val.csv`:

```bash
python flowedit/finetune/prepare_dataset.py \
    --input_dir ./raw_dataset \
    --output_dir ./formatted_dataset
```

### Step 3: Run Fine-Tuning
Run `finetune_xtts.py` to fine-tune the XTTS-v2 GPT weights:

```bash
python flowedit/finetune/finetune_xtts.py \
    --dataset_dir ./formatted_dataset \
    --output_path ./finetune_output \
    --base_model_dir ./flowedit/Xtts \
    --epochs 50 \
    --lr 5e-6
```

### Step 4: Deploy Model Checkpoint
Deploy the fine-tuned `best_model.pth` directly into FlowEdit:

```bash
python flowedit/finetune/deploy_finetuned.py \
    --finetune_dir ./finetune_output \
    --target_dir ./flowedit/Xtts
```

### Step 5: Restart & Verify
Restart the server and verify the output:

```bash
fuser -k 8001/tcp
uvicorn flowedit.api.main:app --host 0.0.0.0 --port 8001 --reload
```
