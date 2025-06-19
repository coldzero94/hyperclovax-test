import gradio as gr
import requests
import time

def stream_chat_api(message, history, image=None):
    url = "http://localhost:8000/infer_stream"
    print(f"[🔁 API 호출] Prompt: {message}, image: {bool(image)}")
    
    # 스트리밍 시작 시간 기록
    start_time = time.time()
    
    data = {"prompt": message}
    files = {}
    if image is not None and image.strip():
        try:
            with open(image, 'rb') as img_file:
                files["image"] = ("image.jpg", img_file.read(), "image/jpeg")
        except Exception as e:
            print(f"이미지 파일 읽기 오류: {e}")
            files = {}
    
    # 스트림 응답을 받아서 yield로 반환
    try:
        with requests.post(url, data=data, files=files if files else None, stream=True) as r:
            partial = ""
            token_count = 0
            is_first_token = True
            first_token_time = 0
            
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    # 첫 토큰 시간 기록
                    if is_first_token:
                        first_token_time = time.time()
                        is_first_token = False
                    
                    partial += chunk
                    token_count += len(chunk.split())  # 간단한 토큰 카운트
                    yield {"role": "assistant", "content": partial + " ⌛"}
            
            # 생성 완료 시간 계산
            end_time = time.time()
            generation_time = end_time - start_time
            first_token_latency = first_token_time - start_time if not is_first_token else 0
            
            # 생성 속도 계산 (초당 토큰)
            if generation_time > 0 and token_count > 0:
                tokens_per_second = token_count / generation_time
            else:
                tokens_per_second = 0
            
            # 생성 통계 추가
            stats = f"\n\n---\n✅ 생성 완료 (토큰: {token_count}개, 시간: {generation_time:.2f}초, 속도: {tokens_per_second:.1f}토큰/초, 첫 토큰: {first_token_latency:.2f}초)"
            
            # 최종 텍스트 반환
            yield {"role": "assistant", "content": partial + stats}
            
    except Exception as e:
        yield {"role": "assistant", "content": f"❌ Error: {str(e)}"}

def main():
    with gr.Blocks(title="🤗 HyperCLOVAX API Chat", fill_height=True) as demo:
        gr.Markdown(
            "<h2>📦 HyperCLOVAX API Chat 📦</h2>"
            "<p>FastAPI 백엔드와 연동된 실시간 채팅 인터페이스입니다.</p>"
        )

        # 상태 표시 추가
        status = gr.Markdown("✨ 준비 완료", elem_id="status")

        # 메시지 형식 사용으로 경고 제거
        chatbot = gr.Chatbot(type="messages", show_copy_button=True, render_markdown=True, scale=20)
        state = gr.State([])  # 메시지 리스트

        with gr.Row():
            txt = gr.Textbox(
                placeholder="메시지를 입력하세요 (Enter: 전송, Shift+Enter: 줄바꿈)",
                lines=2,
                show_label=False,
                scale=8,
                container=False
            )
            image = gr.Image(
                type="filepath", label="이미지 업로드 (선택)", container=True
            )

        with gr.Row():
            send_btn = gr.Button("💬 전송", scale=1)
            retry_btn = gr.Button("🔄 재시도")
            clear_btn = gr.Button("🗑️ 대화 지우기")

        def user_message(message, history, status_text=None):
            """사용자 메시지를 추가하는 함수"""
            if message.strip() == "":
                return "", history, "✨ 준비 완료"
            
            # 이미 history가 있으면 그대로 사용
            if history is None:
                history = []
            
            # 메시지 형식으로 추가
            history.append({"role": "user", "content": message})
            return "", history, "⌛ 응답 생성 중..."
        
        def bot_response(history, status_text=None, img_path=None):
            """봇 응답을 생성하는 함수"""
            if not history:
                yield history, "✨ 준비 완료"
                return
            
            # 마지막 사용자 메시지 가져오기
            last_user_message = history[-1]["content"]
            history_so_far = history[:-1]
            
            # 응답 생성
            for response in stream_chat_api(last_user_message, history_so_far, img_path):
                new_history = history.copy()
                
                # 스트리밍 중인지 완료된건지 확인
                if response["content"].endswith("⌛"):
                    status_update = "⌛ 응답 생성 중..."
                    content = response["content"][:-2]  # 진행 중 표시 제거
                elif "생성 완료" in response["content"]:
                    status_update = "✅ 응답 생성 완료"
                    content = response["content"]
                else:
                    status_update = "✨ 준비 완료"
                    content = response["content"]
                
                # 봇 응답 추가 (메시지 형식)
                if len(new_history) > 0 and new_history[-1]["role"] == "assistant":
                    new_history[-1]["content"] = content
                else:
                    new_history.append({"role": "assistant", "content": content})
                
                yield new_history, status_update

        # 이벤트 연결
        txt.submit(
            user_message, 
            [txt, state, status], 
            [txt, state, status],
            queue=False
        ).then(
            bot_response,
            [state, status, image],
            [chatbot, status]
        )
        
        send_btn.click(
            user_message, 
            [txt, state, status], 
            [txt, state, status],
            queue=False
        ).then(
            bot_response,
            [state, status, image],
            [chatbot, status]
        )

        # 재시도
        def retry_last(history, img_path):
            if not history:
                return history, history, "✨ 준비 완료"
            last_input = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
            trimmed_history = [msg for msg in history if msg["role"] != "assistant"]
            
            def get_retry_response():
                for response in stream_chat_api(last_input, trimmed_history[:-1], img_path):
                    new_history = trimmed_history.copy()
                    
                    # 스트리밍 중인지 완료된건지 확인
                    if response["content"].endswith("⌛"):
                        status_update = "⌛ 응답 생성 중..."
                        content = response["content"][:-2]
                    elif "생성 완료" in response["content"]:
                        status_update = "✅ 응답 생성 완료"
                        content = response["content"]
                    else:
                        status_update = "✨ 준비 완료"
                        content = response["content"]
                    
                    new_history.append({"role": "assistant", "content": content})
                    yield new_history, status_update
            
            return trimmed_history, get_retry_response()

        retry_btn.click(
            fn=retry_last,
            inputs=[state, image],
            outputs=[state, chatbot]
        )

        # 대화 지우기
        clear_btn.click(
            lambda: ([], [], "✨ 준비 완료"), 
            outputs=[chatbot, state, status], 
            queue=False
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