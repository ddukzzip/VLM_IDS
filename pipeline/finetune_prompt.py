import sys, os, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup)
from peft import (LoraConfig, get_peft_model, TaskType,
                  PromptTuningConfig, PromptTuningInit)
from qwen_vl_utils import process_vision_info

DATA_DIR = "/home/oper/VLM_IDS/data"
CKPT_DIR = "/home/oper/VLM_IDS/results/checkpoints_prompt"
LOG_DIR  = "/home/oper/VLM_IDS/results/logs"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

CFG = {
    "lr":               5e-3,   # prompt tuning은 lr 높게
    "epochs":           20,     # 빠른 학습이라 epoch 늘림
    "batch_size":       4,
    "grad_accum":       4,
    "max_len":          512,
    "num_virtual_tokens": 20,   # soft prompt 길이
    "warmup_ratio":     0.1,
}


class DensityQADataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        img = Image.open(
            rec["image"]).convert("RGB").resize(
            (224, 224), Image.BILINEAR)
        messages = [
            {"role": "system",
             "content": (
                 "당신은 네트워크 트래픽 density map을 "
                 "분석하는 IDS 전문가입니다.")},
            {"role": "user",
             "content": [
                 {"type": "image", "image": img},
                 {"type": "text",
                  "text": rec["question"]}]},
            {"role": "assistant",
             "content": rec["answer"]}
        ]
        return messages


def collate_fn(batch):
    return batch


def process_batch(messages_list, processor, model):
    texts = [processor.apply_chat_template(
        m, tokenize=False,
        add_generation_prompt=False)
        for m in messages_list]
    image_inputs, _ = process_vision_info(messages_list)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        truncation=True,
        max_length=CFG["max_len"],
        return_tensors="pt").to(model.device)
    outputs = model(**inputs,
                    labels=inputs["input_ids"])
    return outputs.loss


def train():
    print("모델 로드 중...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)

    prompt_cfg = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=(
            "네트워크 트래픽 density map에서 "
            "benign과 malicious 분포를 분석하고 "
            "noisy label을 탐지하시오."),
        num_virtual_tokens=CFG["num_virtual_tokens"],
        tokenizer_name_or_path=MODEL_ID)
    model = get_peft_model(model, prompt_cfg)
    model.print_trainable_parameters()

    train_ds = DensityQADataset(
        os.path.join(DATA_DIR, "train.json"))
    val_ds   = DensityQADataset(
        os.path.join(DATA_DIR, "val.json"))
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"],
        shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"],
        weight_decay=0.01)
    total_steps  = (len(train_loader) *
                    CFG["epochs"] // CFG["grad_accum"])
    warmup_steps = int(
        total_steps * CFG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    best_val = float("inf")
    log = []

    for epoch in range(CFG["epochs"]):
        # ── train ──────────────────────────────────
        model.train()
        tr_loss = 0.0
        tr_cnt  = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            loss = process_batch(
                batch, processor, model)
            (loss / CFG["grad_accum"]).backward()
            tr_loss += loss.item()
            tr_cnt  += 1

            if (step+1) % CFG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 5 == 0:
                print("  e{} step{}/{} loss={:.4f}".format(
                    epoch+1, step, len(train_loader),
                    tr_loss/tr_cnt))

        # 마지막 gradient
        optimizer.step()
        optimizer.zero_grad()

        # ── val ────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_cnt  = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = process_batch(
                    batch, processor, model)
                val_loss += loss.item()
                val_cnt  += 1

        avg_tr  = tr_loss  / max(1, tr_cnt)
        avg_val = val_loss / max(1, val_cnt)
        print("epoch {}: train={:.4f} val={:.4f}".format(
            epoch+1, avg_tr, avg_val))
        log.append({
            "epoch":      epoch+1,
            "train_loss": avg_tr,
            "val_loss":   avg_val})

        if avg_val < best_val:
            best_val = avg_val
            ckpt = os.path.join(CKPT_DIR, "best_lora")
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            print("  best 저장:", ckpt)

        if (epoch+1) % 5 == 0:
            ckpt = os.path.join(
                CKPT_DIR, "epoch{}".format(epoch+1))
            model.save_pretrained(ckpt)

    with open(os.path.join(
            LOG_DIR, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("완료. best val:", best_val)


if __name__ == "__main__":
    train()
