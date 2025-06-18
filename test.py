from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

app = FastAPI()

# 모델 및 디바이스 설정
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer를 반드시 use_fast=False로 지정
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

# 모델 및 processor 로드
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
model.tokenizer = tokenizer  # 내부에서 자동 호출되는 FastTokenizer를 덮어쓰기

preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

@app.post("/generate")
async def generate(image: UploadFile = File(...), prompt: str = ""):
    # 이미지 로딩 및 RGB 변환
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")

    # 이미지 + 프롬프트 전처리
    inputs = preprocessor(images=img, text=prompt, return_tensors="pt").to(device)

    # 생성 수행
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    # 결과 디코딩
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"response": response}
