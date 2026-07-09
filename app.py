import streamlit as st
import tempfile
import os
import re
import subprocess
from groq import Groq
from dotenv import load_dotenv

# ==========================================
# CẤU HÌNH & KHỞI TẠO
# ==========================================
load_dotenv()

st.set_page_config(page_title="Video Auto-Sub & Translator (Groq Speed)", page_icon="⚡", layout="wide")

# ==========================================
# CÁC HÀM XỬ LÝ LÕI (HELPER FUNCTIONS)
# ==========================================
def extract_audio(video_path: str, audio_path: str):
    """Trích xuất âm thanh dùng FFmpeg, xuất ra MP3 nén để nhẹ dung lượng (<25MB cho Groq)"""
    try:
        # -vn: bỏ video, -acodec libmp3lame: nén mp3, -q:a 5: chất lượng trung bình để giảm size
        command = [
            "ffmpeg", "-i", video_path, 
            "-vn", "-acodec", "libmp3lame", "-q:a", "5", 
            audio_path, "-y"
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Lỗi khi trích xuất âm thanh với FFmpeg. Đảm bảo FFmpeg đã được cài đặt. Chi tiết: {str(e)}")
def format_timestamp(seconds: float) -> str:
    """Định dạng số giây thành chuẩn timecode của SRT: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
def generate_srt_with_groq(client: Groq, audio_path: str) -> str:
    """Sử dụng Groq Whisper API (verbose_json) và tự build thành SRT"""
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), file.read()),
            model="whisper-large-v3",
            response_format="verbose_json", # Đổi từ srt thành verbose_json
        )
    
    srt_content = ""
    
    # Lấy danh sách các câu (segments) từ response
    segments = transcription.segments if hasattr(transcription, 'segments') else transcription.get('segments', [])
    
    # Tự động ghép thành định dạng chuẩn của file SRT
    for i, segment in enumerate(segments, start=1):
        # API trả về có thể là dictionary hoặc object tuỳ phiên bản SDK
        if isinstance(segment, dict):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '')
        else:
            start = getattr(segment, 'start', 0)
            end = getattr(segment, 'end', 0)
            text = getattr(segment, 'text', '')
            
        start_time = format_timestamp(float(start))
        end_time = format_timestamp(float(end))
        
        srt_content += f"{i}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{text.strip()}\n\n"
        
    return srt_content

def translate_srt_with_groq(client: Groq, srt_content: str) -> str:
    """Gọi Groq API để dịch toàn bộ file SRT sang tiếng Việt"""
    prompt = f"""
    Dịch toàn bộ nội dung phụ đề SRT sau đây sang tiếng Việt. 
    YÊU CẦU NGHIÊM NGẶT: 
    1. Giữ nguyên 100% cấu trúc số thứ tự và mốc thời gian (timestamp).
    2. Chỉ dịch phần văn bản hội thoại (text).
    3. Chỉ trả về duy nhất nội dung file SRT đã dịch. 
    4. KHÔNG giải thích, KHÔNG chào hỏi, KHÔNG bọc văn bản trong thẻ markdown (ví dụ: ```srt).
    
    Nội dung file SRT cần dịch:
    {srt_content}
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Bạn là một chuyên gia dịch thuật phụ đề phim chuyên nghiệp. Nhiệm vụ của bạn là dịch chính xác và giữ nguyên cấu trúc SRT."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2, 
    )
    
    translated_text = chat_completion.choices[0].message.content.strip()
    
    if translated_text.startswith("```"):
        lines = translated_text.split('\n')
        if len(lines) > 2:
            translated_text = '\n'.join(lines[1:-1]).strip()
            
    return translated_text

def apply_terminology_regex(text: str, term_dict: dict) -> str:
    """Sử dụng Regex để thay thế thuật ngữ."""
    result_text = text
    for old_word, new_word in term_dict.items():
        pattern = r'\b' + re.escape(old_word) + r'\b'
        result_text = re.sub(pattern, new_word, result_text)
    return result_text

# ==========================================
# GIAO DIỆN STREAMLIT (UI)
# ==========================================
st.title("⚡ AI Subtitle Generator & Translator (Powered by Groq)")
st.markdown("Hệ thống xử lý phụ đề đám mây: Nhận diện và dịch thuật siêu tốc bằng phần cứng Groq LPU.")

if "is_processed" not in st.session_state:
    st.session_state.is_processed = False
if "original_srt" not in st.session_state:
    st.session_state.original_srt = ""
if "translated_srt" not in st.session_state:
    st.session_state.translated_srt = ""

env_api_key = os.getenv("GROQ_API_KEY", "")
api_key = st.text_input("Nhập Groq API Key của bạn (bắt đầu bằng gsk_):", value=env_api_key, type="password")

uploaded_video = st.file_uploader("Tải lên file Video (mp4, mov, avi...)", type=["mp4", "mov", "avi", "mkv"])

st.markdown("### 📝 Tùy chỉnh thuật ngữ (Tùy chọn)")
st.markdown("Nhập các từ cần thay thế sau khi dịch (mỗi dòng 1 cặp, phân cách bằng dấu `=`).")
terminology_input = st.text_area("Danh sách thuật ngữ:", placeholder="vi = vy\nAI = Trí tuệ nhân tạo", height=100)

if uploaded_video and api_key:
    if not st.session_state.is_processed:
        if st.button("🚀 Bắt đầu xử lý với Groq", type="primary", use_container_width=True):
            
            terminology_dict = {}
            if terminology_input.strip():
                for line in terminology_input.split('\n'):
                    if '=' in line:
                        parts = line.split('=')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip()
                            if key and val:
                                terminology_dict[key] = val
            
            client = Groq(api_key=api_key)
            temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            # Đổi đuôi thành mp3 để tối ưu dung lượng cho Groq API
            temp_aud = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_aud.close() 
            
            try:
                temp_vid.write(uploaded_video.read())
                temp_vid.close()
                
                # BƯỚC 1: TÁCH ÂM THANH
                with st.spinner("Đang tách âm thanh từ video bằng FFmpeg... (Bước 1/4)"):
                    extract_audio(temp_vid.name, temp_aud.name)
                
                # Kiểm tra dung lượng file (Groq giới hạn 25MB)
                file_size_mb = os.path.getsize(temp_aud.name) / (1024 * 1024)
                if file_size_mb > 25:
                    st.error(f"❌ Kích thước file âm thanh ({file_size_mb:.2f}MB) vượt quá giới hạn 25MB của Groq API. Vui lòng cắt ngắn video.")
                    st.stop()
                st.success("✅ Đã tách và nén âm thanh thành công!")
                
                # BƯỚC 2: TRÍCH XUẤT PHỤ ĐỀ (GROQ WHISPER API)
                with st.spinner("Đang chạy AI nhận diện giọng nói (Groq Whisper-large-v3)... (Bước 2/4)"):
                    st.session_state.original_srt = generate_srt_with_groq(client, temp_aud.name)
                st.success("✅ Đã tạo phụ đề gốc thành công!")
                
                # BƯỚC 3: DỊCH THUẬT (GROQ LLAMA 3)
                with st.spinner("Đang dịch sang tiếng Việt bằng Groq Llama 3... (Bước 3/4)"):
                    raw_translated_srt = translate_srt_with_groq(client, st.session_state.original_srt)
                st.success("✅ Đã hoàn tất quá trình dịch thuật!")
                
                # BƯỚC 4: ÁP DỤNG TỪ ĐIỂN
                with st.spinner("Đang áp dụng bộ chuẩn hóa thuật ngữ... (Bước 4/4)"):
                    if terminology_dict:
                        st.session_state.translated_srt = apply_terminology_regex(raw_translated_srt, terminology_dict)
                    else:
                        st.session_state.translated_srt = raw_translated_srt
                st.success("✅ Đã hoàn tất xử lý!")
                
                st.session_state.is_processed = True
                st.rerun() 
                
            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi: {str(e)}")
                
            finally:
                if os.path.exists(temp_vid.name):
                    os.remove(temp_vid.name)
                if os.path.exists(temp_aud.name):
                    os.remove(temp_aud.name)

    if st.session_state.is_processed:
        st.markdown("### 📥 Tải xuống kết quả")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(label="Tải file SRT (Gốc)", data=st.session_state.original_srt, file_name="original_subtitles.srt", mime="text/plain", use_container_width=True)
            
        with col2:
            st.download_button(label="Tải file SRT (Tiếng Việt)", data=st.session_state.translated_srt, file_name="translated_subtitles.srt", mime="text/plain", use_container_width=True)
        
        st.markdown("---")
        if st.button("🔄 Bắt đầu lại với video khác", type="secondary", use_container_width=True):
            st.session_state.is_processed = False
            st.session_state.original_srt = ""
            st.session_state.translated_srt = ""
            st.rerun()
                
elif uploaded_video and not api_key:
    st.warning("⚠️ Vui lòng nhập Groq API Key để ứng dụng bắt đầu hoạt động.")