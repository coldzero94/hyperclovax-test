import gradio as gr
import requests

def stream_chat(prompt, image=None):
    url = "http://localhost:8000/infer_stream"
    data = {"prompt": prompt}
    files = {}
    if image is not None:
        files["image"] = ("image.png", image, "image/png")
    # 스트림 응답을 받아서 yield로 반환
    with requests.post(url, data=data, files=files if files else None, stream=True) as r:
        partial = ""
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                partial += chunk
                yield partial

demo = gr.ChatInterface(
    fn=stream_chat,
    textbox=gr.Textbox(placeholder="메시지를 입력하세요", lines=2),
    additional_inputs=[gr.Image(type="filepath", label="이미지 업로드 (선택)")],
    title="HyperCLOVAX Vision Chat (Streaming)",
    description="텍스트와 이미지를 입력하면 실시간으로 답변이 출력됩니다.",
)

demo.launch() 