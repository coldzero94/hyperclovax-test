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
            # 1. Chat 템플릿 구성
            chat = [{"role": "system", "content": "You are a helpful assistant."}]
            for turn in history:
                if turn["role"] != "system":
                    chat.append(turn)
            chat.append({"role": "user", "content": message})

            input_ids = self.tokenizer.apply_chat_template(
                chat, return_tensors="pt", tokenize=True
            ).to(self.device)

            inputs = {"input_ids": input_ids}
            if image:
                image_pil = Image.open(image).convert("RGB")
                pixel_values = self.preprocessor(
                    image_pil, return_tensors="pt"
                ).pixel_values.to(self.device)
                inputs["pixel_values"] = pixel_values

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

            partial = ""
            for new_text in streamer:
                partial += new_text
                yield {"role": "assistant", "content": partial}

        except Exception as e:
            yield {"role": "assistant", "content": f"❌ Error: {str(e)}"}

def main():
    handler = HyperCLOVAXHandler()

    with gr.Blocks(title="🤗 HyperCLOVAX Chat", fill_height=True) as demo:
        gr.Markdown(
            "<h2>🤗 HyperCLOVAX Direct Chat</h2>"
            "<p>Gradio에서 직접 transformers 모델을 사용하는 실시간 채팅 데모입니다.</p>"
        )

        chatbot = gr.Chatbot(type="messages", show_copy_button=True)
        state = gr.State([])  # history as list of {"role": ..., "content": ...}

        with gr.Row():
            txt = gr.Textbox(
                placeholder="메시지를 입력하세요 (Enter: 전송, Shift+Enter: 줄바꿈)",
                lines=2,
                show_label=False
            )
            image = gr.Image(
                type="filepath", label="이미지 업로드 (선택)", container=True
            )

        with gr.Row():
            send_btn = gr.Button("💬 전송")
            retry_btn = gr.Button("🔄 재시도")
            clear_btn = gr.Button("🗑️ 대화 지우기")

        def user_submit(message, history, img_path):
            history = history + [{"role": "user", "content": message}]
            return "", history, handler.stream_chat(message, history, img_path)

        send_btn.click(
            fn=user_submit,
            inputs=[txt, state, image],
            outputs=[txt, state, chatbot]
        )

        def retry_last(history, img_path):
            if not history:
                return history, chatbot
            last_input = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
            trimmed_history = [msg for msg in history if msg["role"] != "assistant"]
            trimmed_history.append({"role": "user", "content": last_input})
            return trimmed_history, handler.stream_chat(last_input, trimmed_history, img_path)

        retry_btn.click(
            fn=retry_last,
            inputs=[state, image],
            outputs=[state, chatbot]
        )

        clear_btn.click(
            lambda: ([], []),
            outputs=[chatbot, state]
        )

    demo.queue().launch(server_name="0.0.0.0", server_port=7861)

if __name__ == "__main__":
    main()
