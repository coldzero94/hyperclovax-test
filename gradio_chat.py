import gradio as gr
import requests

def stream_chat(message, history, image=None):
    url = "http://localhost:8000/infer_stream"
    print(f"[🔁 API 호출] Prompt: {message}, image: {bool(image)}")
    
    data = {"prompt": message}
    files = {}
    if image is not None:
        files["image"] = ("image.png", image, "image/png")
    
    # 스트림 응답을 받아서 yield로 반환
    try:
        with requests.post(url, data=data, files=files if files else None, stream=True) as r:
            partial = ""
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    partial += chunk
                    yield {"role": "assistant", "content": partial}
    except Exception as e:
        yield {"role": "assistant", "content": f"❌ Error: {str(e)}"}

def main():
    with gr.Blocks(title="🤗 HyperCLOVAX API Chat", fill_height=True) as demo:
        gr.Markdown(
            "<h2>🤗 HyperCLOVAX API Chat</h2>"
            "<p>FastAPI 백엔드와 연동된 실시간 채팅 인터페이스입니다.</p>"
        )

        chatbot = gr.Chatbot(type="messages", show_copy_button=True)
        state = gr.State([])  # 메시지 리스트

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

        # 사용자 입력 처리
        def user_submit(message, history, img_path):
            if not message.strip():
                return "", history, history
            history = history + [{"role": "user", "content": message}]
            return "", history, stream_chat(message, history, img_path)

        send_btn.click(
            fn=user_submit,
            inputs=[txt, state, image],
            outputs=[txt, state, chatbot]
        )

        txt.submit(  # 엔터 키로도 전송
            fn=user_submit,
            inputs=[txt, state, image],
            outputs=[txt, state, chatbot]
        )

        # 재시도
        def retry_last(history, img_path):
            if not history:
                return history, history
            # 마지막 사용자 메시지 찾기
            last_input = ""
            for msg in reversed(history):
                if msg["role"] == "user":
                    last_input = msg["content"]
                    break
            
            if not last_input:
                return history, history
                
            # assistant 메시지들 제거하고 다시 생성
            trimmed_history = [msg for msg in history if msg["role"] == "user"]
            return trimmed_history, stream_chat(last_input, trimmed_history, img_path)

        retry_btn.click(
            fn=retry_last,
            inputs=[state, image],
            outputs=[state, chatbot]
        )

        # 대화 지우기
        def clear_chat():
            return [], []

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, state]
        )

        # chatbot 상태를 state에 동기화
        def sync_chatbot_to_state(chatbot_history):
            return chatbot_history

        chatbot.change(
            fn=sync_chatbot_to_state,
            inputs=[chatbot],
            outputs=[state]
        )

    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main() 