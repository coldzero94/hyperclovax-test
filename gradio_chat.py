import gradio as gr
import requests

def stream_chat(message, history, image=None):
    url = "http://localhost:8000/infer_stream"
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
                    yield partial
    except Exception as e:
        yield f"Error: {str(e)}"

def main():
    with gr.Blocks(title="🤗 HyperCLOVAX Vision Chat", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                "<h2>🤗 HyperCLOVAX Vision Chat (Streaming) 🤗</h2>"
                "<h3>텍스트와 이미지를 입력하면 실시간으로 답변이 출력됩니다!<br></h3>"
                "<h3>FastAPI 백엔드와 연동된 채팅 인터페이스</h3>")
        
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
            fn=stream_chat,
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

    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main() 