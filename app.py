import streamlit as st
import openai
import os
import io
from pydub import AudioSegment
from dotenv import load_dotenv

# --- 1. 설정 및 API 클라이언트 초기화 ---
# OPENAI_API_KEY는 Codespaces Secret에서 자동으로 로드됩니다.
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY:
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        st.session_state.client = client
    except Exception as e:
        st.error(f"OpenAI 클라이언트 초기화 오류: {e}")
        st.stop()
else:
    st.error("오류: OPENAI_API_KEY가 설정되지 않았습니다. GitHub Secrets를 확인해 주세요.")
    st.stop()

# --- 2. LLM 프롬프트 템플릿 정의 ---

TONE_PROMPT_TEMPLATE = """
당신은 최고의 스피치 코치입니다. 당신의 목표는 사용자에게 {purpose}에 최적화된 발화 및 형식 피드백을 제공하는 것입니다.

[입력 데이터]
1. 목표 청중/상황: {purpose}
2. 발화 스크립트: "{script}"
3. 측정된 발화 속도: 분당 {wpm} 단어

[요구사항]
1. [발화 속도] {purpose}에 적합한 표준 WPM을 제시하고, 현재 {wpm}이 적절한지 구체적으로 진단하고 개선 방향을 제시하십시오.
2. [톤앤매너 및 어휘] 스크립트를 분석하여 {purpose}에 부적합한 구어체, 모호한 표현, 반복되거나 비전문적인 어휘 5개 이상을 지적하고, 이를 대체할 전문적인 어휘나 문장 구조를 추천하십시오.

출력은 반드시 Markdown 형식으로 작성하며, 각 섹션에 명확한 소제목을 붙여주십시오.
"""

LOGIC_PROMPT_TEMPLATE = """
당신은 논리 컨설팅 전문가이며, 청중의 질문을 예측하는 훈련된 전략가입니다.
당신의 목표는 스크립트의 논리적 결함을 찾아내고, 질의응답을 완벽하게 대비시키는 것입니다.

[입력 데이터]
1. 목표 청중/상황: {purpose}
2. 발화 스크립트: "{script}"

[요구사항]
1. [논리 결함 진단] 스크립트 내용을 비판적으로 분석하여, 청중이 의문을 가질 만한 논리적 비약, 근거 부족, 주장의 모호성 등 핵심 약점 3가지를 찾으십시오. 각 약점은 스크립트 내 해당 부분을 인용하여 명확히 설명하십시오.
2. [예상 꼬리 질문] 진단된 3가지 논리적 약점 각각을 파고드는, {purpose}에 적합한 날카로운 꼬리 질문(Follow-up Questions) 3개를 생성하십시오. (총 9개의 질문)
3. [개선 방안] 논리 결함을 해소하기 위해 스크립트에 추가해야 할 구체적인 데이터 유형이나 설명 요소를 제시하십시오.

출력은 반드시 Markdown 형식으로 작성하며, 각 섹션에 명확한 소제목을 붙여주십시오.
"""

# --- 4. STT 및 WPM 계산 함수 ---

def process_audio(audio_bytes, filename):
    """Whisper STT 변환 및 WPM/길이 계산 통합 함수"""
    
    # 임시 파일 경로
    temp_path = f"/tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    
    try:
        # Whisper API 호출
        with open(temp_path, "rb") as audio_file:
            transcript_response = st.session_state.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text"
            )
        transcript = transcript_response
        
        # 오디오 길이 측정 및 WPM 계산
        audio = AudioSegment.from_file(temp_path)
        total_time_minutes = len(audio) / 1000 / 60
        word_count = len(transcript.split())
        wpm = round(word_count / total_time_minutes) if total_time_minutes > 0 else 0
        
        os.remove(temp_path) # 임시 파일 정리
        
        return transcript, wpm, total_time_minutes, word_count
        
    except Exception as e:
        st.error(f"오디오 처리/STT 변환 중 오류 발생: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return None, 0, 0, 0

def get_llm_feedback(script, purpose, wpm):
    """LLM을 호출하여 두 가지 피드백을 생성"""
    try:
        tone_prompt = TONE_PROMPT_TEMPLATE.format(purpose=purpose, script=script, wpm=wpm)
        tone_response = st.session_state.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": tone_prompt}]
        )
        tone_feedback = tone_response.choices[0].message.content

        logic_prompt = LOGIC_PROMPT_TEMPLATE.format(purpose=purpose, script=script)
        logic_response = st.session_state.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": logic_prompt}]
        )
        logic_feedback = logic_response.choices[0].message.content

        return tone_feedback, logic_feedback

    except Exception as e:
        st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
        return None, None

# --- 5. Streamlit UI 및 메인 실행 로직 ---

def main():
    st.set_page_config(page_title="Prep Master: AI 스피치 코치", layout="wide")
    st.title("🎤 Prep Master: AI 스피치 코치 (STT 통합 버전)")
    st.markdown("### 녹음 파일을 업로드하면 AI가 대본을 추출하고 분석합니다.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        purpose = st.selectbox(
            "1. 발표/면접 목적을 선택하세요:",
            ["IR 피치 (투자)", "취업 면접 (전문직)", "학술 발표 (논문)", "일반 팀 발표"]
        )
        uploaded_file = st.file_uploader(
            "2. 연습 녹음 파일 (.mp3, .wav, .m4a)을 업로드하세요.",
            type=["mp3", "wav", "m4a"]
        )

    analyze_button = st.button("🚀 AI 코칭 시작!", use_container_width=True)
    st.markdown("---")

    if analyze_button:
        if not uploaded_file:
            st.error("오디오 파일을 업로드해 주세요.")
            st.stop()
        
        with st.spinner('⏳ [1/2단계] 오디오 분석 및 대본 추출 중 (Whisper API 호출)...'):
            audio_bytes = uploaded_file.read()
            filename = uploaded_file.name
            
            transcript, wpm, total_time, word_count = process_audio(audio_bytes, filename)
            
            if not transcript:
                st.stop()

        with col2:
             st.text_area("🔍 Whisper가 추출한 대본", transcript, height=300, disabled=True)
             
        
        with st.spinner('🧠 [2/2단계] AI가 내용과 발화를 분석 중입니다...'):
            tone_feedback, logic_feedback = get_llm_feedback(transcript, purpose, wpm)
            
        if tone_feedback and logic_feedback:
            st.success("🎉 분석 완료! 아래에서 피드백을 확인하세요.")
            
            tab1, tab2 = st.tabs(["🗣️ 발화 & 형식 피드백", "🧠 내용 & 논리 피드백"])
            
            with tab1:
                st.subheader("📊 발화 속도 분석")
                
                STANDARD_MIN = 120
                STANDARD_MAX = 160
                
                st.metric(
                    label="측정된 발화 속도 (WPM)", 
                    value=f"{wpm}", 
                    delta=f"{total_time:.1f}분 동안 {word_count}단어 발화"
                )
                
                color = 'green'
                if wpm < STANDARD_MIN:
                    status_msg = "느림"
                    color = 'orange'
                elif wpm > STANDARD_MAX:
                    status_msg = "빠름"
                    color = 'red'
                else:
                    status_msg = "적정"

                st.markdown(f"**속도 평가:** :bulb: <span style='color:{color}'>{status_msg}</span>", unsafe_allow_html=True)
                st.progress(min(wpm / 200.0, 1.0))
                st.markdown("---")
                
                st.subheader("🗣️ AI 스피치 코치 피드백")
                st.markdown(tone_feedback)
                
            with tab2:
                st.markdown(logic_feedback)

if __name__ == "__main__":
    main()