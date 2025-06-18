from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from typing import List, Union
import requests
from PIL import Image
from io import BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HyperCLOVAX API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, tokenizer, processor
MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {device}")
logger.info(f"Loading model: {MODEL_NAME}")

try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Define chat schema
class ChatMessage(BaseModel):
    role: str
    content: Union[str, dict]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def root():
    return {"message": "HyperCLOVAX API is running", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "healthy", "device": device}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Apply chat template
        chat = [msg.dict() for msg in request.messages]
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True).to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.6,
                temperature=0.5,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Extract only the new generated text
        input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        response_text = decoded[len(input_text):].strip()
        
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vlm-chat", response_model=ChatResponse)
def vlm_chat(request: ChatRequest):
    try:
        chat = [msg.dict() for msg in request.messages]

        # Load images/videos from URL
        new_chat, all_images, is_video_list = processor.load_images_videos(chat)
        preprocessed = processor(all_images, is_video_list=is_video_list)

        input_ids = tokenizer.apply_chat_template(
            new_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=8192,
                do_sample=True,
                top_p=0.6,
                temperature=0.5,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                **preprocessed,
            )
        
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Extract only the new generated text
        input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        response_text = decoded[len(input_text):].strip()
        
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error in vlm-chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
