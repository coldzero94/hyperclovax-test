from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import io

app = FastAPI()

# 모델 로딩 (vision+language 모델은 AutoModelForVision2Seq 형태)
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

@app.post("/generate")
async def generate(image: UploadFile = File(...), prompt: str = ""):
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    inputs = processor(prompt=prompt, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"response": text}
