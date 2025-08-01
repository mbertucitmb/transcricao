import streamlit as st
import subprocess
import sys
import tempfile
import os
import io
import numpy as np
import time
from pathlib import Path

# Função para instalar pacotes automaticamente
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Verificação e instalação automática de dependências
def check_and_install_dependencies():
    required_packages = {
        'whisper': 'openai-whisper',
        'pydub': 'pydub',
        'torch': 'torch',
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        with st.spinner(f"Instalando dependências: {', '.join(missing_packages)}..."):
            for package in missing_packages:
                try:
                    install_package(package)
                    st.success(f"✅ {package} instalado com sucesso!")
                except Exception as e:
                    st.error(f"❌ Erro ao instalar {package}: {str(e)}")
                    return False
        st.rerun()
    return True

# Verifica dependências na inicialização
if not check_and_install_dependencies():
    st.stop()

# Agora importa os módulos necessários
try:
    import whisper
    from pydub import AudioSegment
    import torch
except ImportError as e:
    st.error(f"Erro ao importar módulos: {str(e)}")
    st.stop()

# Configuração da página
st.set_page_config(
    page_title="Transcritor de Áudio",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache do modelo para evitar recarregamento
@st.cache_resource
def load_whisper_model(model_size):
    """Carrega o modelo Whisper especificado"""
    try:
        model = whisper.load_model(model_size)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None

def convert_audio_format(audio_file, target_format="wav"):
    """Converte áudio para formato compatível"""
    try:
        # Lê o arquivo de áudio
        audio = AudioSegment.from_file(audio_file)
        
        # Converte para mono e 16kHz (otimizado para Whisper)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Salva em formato temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{target_format}") as tmp_file:
            audio.export(tmp_file.name, format=target_format)
            return tmp_file.name
    except Exception as e:
        # Fallback: tenta usar o arquivo original se a conversão falhar
        st.warning(f"Conversão falhou, usando arquivo original: {str(e)}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.getvalue() if hasattr(audio_file, 'getvalue') else audio_file.read())
            return tmp_file.name

def transcribe_audio(model, audio_path, language=None, task="transcribe"):
    """Transcreve o áudio usando Whisper"""
    try:
        # Opções de transcrição
        options = {
            "task": task,
            "fp16": torch.cuda.is_available(),  # Usa FP16 se GPU disponível
        }
        
        if language and language != "auto":
            options["language"] = language
            
        # Realiza a transcrição
        result = model.transcribe(audio_path, **options)
        return result
    except Exception as e:
        st.error(f"Erro na transcrição: {str(e)}")
        return None

def format_timestamp(seconds):
    """Formata timestamp em formato legível"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    st.title("🎤 Transcritor de Áudio Robusto")
    st.markdown("### Transcrição precisa usando Whisper (OpenAI)")
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Seleção do modelo
        model_size = st.selectbox(
            "Modelo Whisper:",
            ["tiny", "base", "small", "medium", "large"],
            index=2,
            help="Modelos maiores são mais precisos, mas mais lentos"
        )
        
        # Idioma
        languages = {
            "auto": "Detecção Automática",
            "pt": "Português",
            "en": "Inglês",
            "es": "Espanhol",
            "fr": "Francês",
            "de": "Alemão",
            "it": "Italiano",
            "ja": "Japonês",
            "ko": "Coreano",
            "zh": "Chinês"
        }
        
        selected_language = st.selectbox(
            "Idioma do áudio:",
            list(languages.keys()),
            format_func=lambda x: languages[x]
        )
        
        # Tipo de tarefa
        task = st.radio(
            "Tarefa:",
            ["transcribe", "translate"],
            format_func=lambda x: "Transcrever" if x == "transcribe" else "Traduzir para inglês"
        )
        
        # Opções de saída
        st.subheader("📄 Opções de Saída")
        show_timestamps = st.checkbox("Mostrar timestamps", value=True)
        show_confidence = st.checkbox("Mostrar nível de confiança", value=False)
        
    # Interface principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload do Áudio")
        
        # Upload do arquivo
        uploaded_file = st.file_uploader(
            "Escolha um arquivo de áudio",
            type=['wav', 'mp3', 'mp4', 'avi', 'mov', 'mkv', 'flac', 'aac', 'm4a'],
            help="Formatos suportados: WAV, MP3, MP4, AVI, MOV, MKV, FLAC, AAC, M4A"
        )
        
        if uploaded_file:
            st.success(f"Arquivo carregado: {uploaded_file.name}")
            
            # Informações do arquivo
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.info(f"Tamanho: {file_size:.2f} MB")
            
            # Player de áudio
            st.audio(uploaded_file, format='audio/wav')
    
    with col2:
        st.subheader("🔄 Status da Transcrição")
        
        if uploaded_file:
            if st.button("🚀 Iniciar Transcrição", type="primary"):
                # Barra de progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Carrega o modelo
                    status_text.text("Carregando modelo Whisper...")
                    progress_bar.progress(20)
                    model = load_whisper_model(model_size)
                    
                    if model is None:
                        st.error("Falha ao carregar o modelo")
                        return
                    
                    # Converte o áudio
                    status_text.text("Processando áudio...")
                    progress_bar.progress(40)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        converted_path = convert_audio_format(tmp_file.name)
                    
                    if converted_path is None:
                        st.error("Falha na conversão do áudio")
                        return
                    
                    # Transcreve
                    status_text.text("Transcrevendo áudio...")
                    progress_bar.progress(60)
                    
                    start_time = time.time()
                    result = transcribe_audio(
                        model, 
                        converted_path, 
                        language=selected_language if selected_language != "auto" else None,
                        task=task
                    )
                    end_time = time.time()
                    
                    if result is None:
                        st.error("Falha na transcrição")
                        return
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Transcrição concluída!")
                    
                    # Limpa arquivos temporários
                    os.unlink(converted_path)
                    
                    # Exibe resultados
                    st.success(f"Transcrição concluída em {end_time - start_time:.2f} segundos")
                    
                    # Idioma detectado
                    if "language" in result:
                        detected_lang = result["language"]
                        st.info(f"Idioma detectado: {detected_lang}")
                    
                except Exception as e:
                    st.error(f"Erro durante a transcrição: {str(e)}")
                    return
    
    # Área de resultados
    if uploaded_file and 'result' in locals():
        st.markdown("---")
        st.subheader("📝 Resultado da Transcrição")
        
        # Texto completo
        with st.expander("📄 Texto Completo", expanded=True):
            full_text = result["text"].strip()
            st.text_area("", value=full_text, height=200, key="full_text")
            
            # Botão de download
            st.download_button(
                label="📥 Download Texto",
                data=full_text,
                file_name=f"transcricao_{uploaded_file.name}.txt",
                mime="text/plain"
            )
        
        # Segmentos com timestamps
        if show_timestamps and "segments" in result:
            with st.expander("⏱️ Transcrição com Timestamps"):
                for i, segment in enumerate(result["segments"]):
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    
                    if show_confidence and "avg_logprob" in segment:
                        # Converte log probability para porcentagem aproximada
                        confidence = np.exp(segment["avg_logprob"]) * 100
                        st.write(f"**[{start_time} - {end_time}]** ({confidence:.1f}%): {text}")
                    else:
                        st.write(f"**[{start_time} - {end_time}]**: {text}")
        
        # Estatísticas
        if "segments" in result:
            with st.expander("📊 Estatísticas"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Duração Total", f"{len(result['segments'])} segmentos")
                
                with col2:
                    total_duration = result["segments"][-1]["end"] if result["segments"] else 0
                    st.metric("Duração", f"{total_duration:.1f}s")
                
                with col3:
                    word_count = len(full_text.split())
                    st.metric("Palavras", word_count)

    # Informações sobre o sistema
    with st.sidebar:
        st.markdown("---")
        st.subheader("ℹ️ Informações do Sistema")
        
        # Verifica GPU
        if torch.cuda.is_available():
            st.success("🚀 GPU disponível (CUDA)")
            st.write(f"GPU: {torch.cuda.get_device_name()}")
        else:
            st.info("💻 Usando CPU")
        
        st.markdown("---")
        st.markdown("**Tecnologias utilizadas:**")
        st.markdown("- 🤖 OpenAI Whisper")
        st.markdown("- 🎵 PyDub para processamento")
        st.markdown("- 🔥 PyTorch para ML")
        st.markdown("- 🌟 Streamlit para interface")

if __name__ == "__main__":
    main()
