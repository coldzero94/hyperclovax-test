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
    if image:
        # 이미지 업로드 시 Vision + Text 모드
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = preprocessor(
            images=image_pil,
            text=prompt,
            return_tensors="pt"
        ).to(device)
    else:
        # 텍스트만 있을 때는 None을 명시
        inputs = preprocessor(
            images=None,
            text=prompt,
            return_tensors="pt"
        ).to(device)

    outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"result": response}
