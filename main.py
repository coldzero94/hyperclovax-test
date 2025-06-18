from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

app = FastAPI()

# 모델 로딩
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# trust_remote_code와 use_fast=False로 tokenizer 오류 방지
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
).to(device)

preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False
)

@app.post("/generate")
async def generate(image: UploadFile = File(...), prompt: str = ""):
    # 이미지 로딩
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")

    # 전처리 및 tensor 변환
    inputs = preprocessor(images=img, text=prompt, return_tensors="pt").to(device)

    # 텍스트 생성
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    # 결과 디코딩
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"response": response}
