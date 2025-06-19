from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from PIL import Image
import io
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer
import threading

app = FastAPI()

# 모델 로딩
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.post("/infer")
async def infer(prompt: str = Form(...), image: UploadFile = File(None)):
    print(f"🔁 일반 요청: prompt='{prompt[:50]}...', image={bool(image)}")
    
    try:
        # HyperCLOVAX 올바른 방식으로 채팅 구성
        vlm_chat = [
            {"role": "system", "content": {"type": "text", "text": "You are a helpful assistant."}},
            {"role": "user", "content": {"type": "text", "text": prompt}}
        ]
        
        # 이미지가 있으면 VLM 방식으로 추가
        if image:
            try:
                image_bytes = await image.read()
                print(f"📷 이미지 파일 크기: {len(image_bytes)} bytes")
                
                if len(image_bytes) > 0:
                    # 임시 파일로 저장 (HyperCLOVAX가 파일 경로나 PIL 객체 필요)
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(image_bytes)
                        tmp_path = tmp_file.name
                    
                    # HyperCLOVAX 방식으로 이미지 추가
                    vlm_chat.append({
                        "role": "user",
                        "content": {
                            "type": "image",
                            "filename": "uploaded_image.jpg",
                            "image": tmp_path,
                        }
                    })
                    print("✅ VLM 채팅에 이미지 추가 완료")
                    
                    # HyperCLOVAX 올바른 전처리 방식
                    new_vlm_chat, all_images, is_video_list = preprocessor.load_images_videos(vlm_chat)
                    preprocessed = preprocessor(all_images, is_video_list=is_video_list)
                    
                    # 채팅 템플릿 적용
                    input_ids = tokenizer.apply_chat_template(
                        new_vlm_chat, 
                        return_tensors="pt", 
                        tokenize=True, 
                        add_generation_prompt=True
                    ).to(device)
                    
                    print("✅ HyperCLOVAX VLM 전처리 완료")
                    
                    # 임시 파일 정리
                    import os
                    os.unlink(tmp_path)
                    
                else:
                    print("⚠️ 빈 이미지 파일, 텍스트만 처리")
                    # 텍스트만 처리
                    input_ids = tokenizer.apply_chat_template(
                        vlm_chat, 
                        return_tensors="pt", 
                        tokenize=True, 
                        add_generation_prompt=True
                    ).to(device)
                    preprocessed = {}
                    
            except Exception as img_error:
                print(f"❌ 이미지 처리 오류: {str(img_error)}")
                # 이미지 실패 시 텍스트만 처리
                input_ids = tokenizer.apply_chat_template(
                    vlm_chat, 
                    return_tensors="pt", 
                    tokenize=True, 
                    add_generation_prompt=True
                ).to(device)
                preprocessed = {}
        else:
            # 텍스트만 처리
            input_ids = tokenizer.apply_chat_template(
                vlm_chat, 
                return_tensors="pt", 
                tokenize=True, 
                add_generation_prompt=True
            ).to(device)
            preprocessed = {}
            print("✅ 텍스트만 처리")

        # 생성
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.6,
            temperature=0.5,
            repetition_penalty=1.0,
            **preprocessed  # HyperCLOVAX 전처리 결과 추가
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"result": response}
        
    except Exception as e:
        print(f"❌ 일반 생성 오류: {str(e)}")
        return {"result": f"Error: {str(e)}"}

@app.post("/infer_stream")
async def infer_stream(prompt: str = Form(...), image: UploadFile = File(None)):
    print(f"🔁 스트림 요청: prompt='{prompt[:50]}...', image={bool(image)}")
    
    try:
        # HyperCLOVAX 올바른 방식으로 채팅 구성
        vlm_chat = [
            {"role": "system", "content": {"type": "text", "text": "You are a helpful assistant."}},
            {"role": "user", "content": {"type": "text", "text": prompt}}
        ]
        
        # 이미지가 있으면 VLM 방식으로 추가
        if image:
            try:
                image_bytes = await image.read()
                print(f"📷 이미지 파일 크기: {len(image_bytes)} bytes")
                
                if len(image_bytes) > 0:
                    # 임시 파일로 저장 (HyperCLOVAX가 파일 경로나 PIL 객체 필요)
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(image_bytes)
                        tmp_path = tmp_file.name
                    
                    # HyperCLOVAX 방식으로 이미지 추가
                    vlm_chat.append({
                        "role": "user",
                        "content": {
                            "type": "image",
                            "filename": "uploaded_image.jpg",
                            "image": tmp_path,
                        }
                    })
                    print("✅ VLM 채팅에 이미지 추가 완료")
                    
                    # HyperCLOVAX 올바른 전처리 방식
                    new_vlm_chat, all_images, is_video_list = preprocessor.load_images_videos(vlm_chat)
                    preprocessed = preprocessor(all_images, is_video_list=is_video_list)
                    
                    # 채팅 템플릿 적용
                    input_ids = tokenizer.apply_chat_template(
                        new_vlm_chat, 
                        return_tensors="pt", 
                        tokenize=True, 
                        add_generation_prompt=True
                    ).to(device)
                    
                    print("✅ HyperCLOVAX VLM 전처리 완료")
                    
                    # 임시 파일 정리
                    import os
                    os.unlink(tmp_path)
                    
                else:
                    print("⚠️ 빈 이미지 파일, 텍스트만 처리")
                    # 텍스트만 처리
                    input_ids = tokenizer.apply_chat_template(
                        vlm_chat, 
                        return_tensors="pt", 
                        tokenize=True, 
                        add_generation_prompt=True
                    ).to(device)
                    preprocessed = {}
                    
            except Exception as img_error:
                print(f"❌ 이미지 처리 오류: {str(img_error)}")
                # 이미지 실패 시 텍스트만 처리
                input_ids = tokenizer.apply_chat_template(
                    vlm_chat, 
                    return_tensors="pt", 
                    tokenize=True, 
                    add_generation_prompt=True
                ).to(device)
                preprocessed = {}
        else:
            # 텍스트만 처리
            input_ids = tokenizer.apply_chat_template(
                vlm_chat, 
                return_tensors="pt", 
                tokenize=True, 
                add_generation_prompt=True
            ).to(device)
            preprocessed = {}
            print("✅ 텍스트만 처리")

        # 스트리밍 생성
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.6,
            temperature=0.5,
            repetition_penalty=1.0,
            streamer=streamer,
            **preprocessed  # HyperCLOVAX 전처리 결과 추가
        )
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def token_stream():
            for new_text in streamer:
                yield new_text

        return StreamingResponse(token_stream(), media_type="text/plain")
        
    except Exception as e:
        print(f"❌ 스트림 생성 오류: {str(e)}")
        def error_stream():
            yield f"Error: {str(e)}"
        return StreamingResponse(error_stream(), media_type="text/plain")
