import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextIteratorStreamer
from PIL import Image
import threading

class HyperCLOVAXHandler:
    def __init__(self):
        print("🚀 모델 로딩 중...")
        self.model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        ).to(self.device)
        self.preprocessor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("✅ 모델 로딩 완료!")

    def stream_chat(self, message, history, image=None):
        try:
            # 1. Chat template 구성
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
            input_ids = self.tokenizer.apply_chat_template(
                chat, return_tensors="pt", tokenize=True
            ).to(self.device)

            # 2. 이미지 처리 (있을 경우)
            inputs = {"input_ids": input_ids}
            if image:
                image_pil = Image.open(image).convert("RGB")
                pixel_values = self.preprocessor(
                    image_pil, return_tensors="pt"
                ).pixel_values.to(self.device)
                inputs["pixel_values"] = pixel_values

            # 3. 스트리밍 생성
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_special_tokens=True
            )
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.6,
                temperature=0.5,
                repetition_penalty=1.0,
                streamer=streamer,
            )
            
            thread = threading.Thread(
                target=self.model.generate, 
                kwargs=generation_kwargs
            )
            thread.start()

            # 4. 스트림 응답
            partial = ""
            for new_text in streamer:
                partial += new_text
                yield partial
                
        except Exception as e:
            yield f"❌ Error: {str(e)}"

def main():
    # 모델 핸들러 초기화
    handler = HyperCLOVAXHandler()
    
    with gr.Blocks(title="🤗 HyperCLOVAX Direct Chat", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                "<h2>🤗 HyperCLOVAX Direct Chat (No API) 🤗</h2>"
                "<h3>Gradio에서 직접 transformers 모델 사용!</h3>"
                "<h3>✨ FastAPI 없이 직접 연결된 채팅 인터페이스 ✨</h3>"
            )
        
        # 커스텀 챗봇 위젯
        chatbot = gr.Chatbot(
            type="messages", 
            scale=20, 
            render_markdown=True,
            show_label=False,
            container=True,
            show_copy_button=True
        )
        
        # 채팅 인터페이스
        gr.ChatInterface(
            fn=handler.stream_chat,
            chatbot=chatbot,
            textbox=gr.Textbox(
                placeholder="메시지를 입력하세요 (Enter: 전송, Shift+Enter: 줄바꿈)", 
                lines=2,
                show_label=False
            ),
            additional_inputs=[
                gr.Image(
                    type="filepath", 
                    label="이미지 업로드 (선택)",
                    container=True
                )
            ],
            submit_btn="💬 전송",
            retry_btn="🔄 재시도", 
            undo_btn="↩️ 되돌리기",
            clear_btn="🗑️ 대화 지우기"
        )

    demo.queue().launch(server_name="0.0.0.0", server_port=7861)

if __name__ == "__main__":
    main() 