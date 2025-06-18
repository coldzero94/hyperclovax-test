from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from PIL import Image
import io

app = FastAPI()

# 모델 로딩
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.post("/infer")
async def infer(prompt: str = Form(...), image: UploadFile = File(None)):
    # 1. Chat template 구성
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True).to(device)

    # 2. 이미지 처리 (있을 경우)
    inputs = {"input_ids": input_ids}
    if image:
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = preprocessor(image_pil, return_tensors="pt").pixel_values.to(device)
        inputs["pixel_values"] = pixel_values

    # 3. 생성
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.6,
        temperature=0.5,
        repetition_penalty=1.0,
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"result": response}
