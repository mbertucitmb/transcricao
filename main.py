import streamlit as st
import speech_recognition as sr
import tempfile
import os
import io
import numpy as np
import time
from pathlib import Path
import subprocess
import sys

# Função para instalar pacotes automaticamente
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Verificação e instalação automática de dependências
def check_and_install_dependencies():
    required_packages = {
        'speech_recognition': 'SpeechRecognition',
        'pydub': 'pydub',
        'pyaudio': 'pyaudio',
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
                    if package == 'pyaudio':
                        # PyAudio pode ser problemático no Windows, tenta alternativa
                        try:
                            install_package('pyaudio')
                        except:
                            st.warning("PyAudio não instalado - algumas funcionalidades podem ser limitadas")
                            continue
                    else:
                        install_package(package)
                    st.success(f"✅ {package} instalado com sucesso!")
                except Exception as e:
                    st.error(f"❌ Erro ao instalar {package}: {str(e)}")
        st.rerun()
    return True

# Verifica dependências na inicialização
if not check_and_install_dependencies():
    st.stop()

# Agora importa os módulos necessários
try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
except ImportError as e:
    st.error(f"Erro ao importar módulos de áudio: {str(e)}")
    st.stop()

# Configuração da página
st.set_page_config(
    page_title="Transcritor de Áudio Robusto",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache do reconhecedor para evitar recarregamento
@st.cache_resource
def get_speech_recognizer():
    """Inicializa o reconhecedor de fala"""
    return sr.Recognizer()

def convert_audio_to_wav(audio_file):
    """Converte áudio para formato WAV compatível"""
    try:
        # Lê o arquivo de áudio
        audio = AudioSegment.from_file(audio_file)
        
        # Converte para mono e taxa de amostragem padrão
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Salva em formato temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio.export(tmp_file.name, format="wav")
            return tmp_file.name, len(audio) / 1000.0  # duração em segundos
    except Exception as e:
        st.error(f"Erro na conversão de áudio: {str(e)}")
        return None, 0

def transcribe_with_google(recognizer, audio_file, language="pt-BR"):
    """Transcreve usando Google Speech Recognition (gratuito)"""
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        
        # Tenta transcrever
        text = recognizer.recognize_google(audio_data, language=language)
        return text, "Google Speech Recognition"
    except sr.UnknownValueError:
        return "Não foi possível entender o áudio", "Google Speech Recognition"
    except sr.RequestError as e:
        return f"Erro na solicitação: {str(e)}", "Google Speech Recognition"
    except Exception as e:
        return f"Erro: {str(e)}", "Google Speech Recognition"

def transcribe_with_sphinx(recognizer, audio_file):
    """Transcreve usando PocketSphinx (offline)"""
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        
        text = recognizer.recognize_sphinx(audio_data)
        return text, "PocketSphinx (Offline)"
    except sr.UnknownValueError:
        return "Não foi possível entender o áudio", "PocketSphinx (Offline)"
    except Exception as e:
        return f"Erro: {str(e)}", "PocketSphinx (Offline)"

def transcribe_large_audio(recognizer, audio_file, engine="google", language="pt-BR", chunk_length=30):
    """Transcreve áudios longos dividindo em chunks"""
    try:
        # Carrega o áudio
        audio = AudioSegment.from_wav(audio_file)
        
        # Divide em chunks de silêncio ou por tempo
        chunks = split_on_silence(
            audio,
            min_silence_len=1000,  # 1 segundo de silêncio
            silence_thresh=audio.dBFS-14,
            keep_silence=500,
        )
        
        # Se não conseguiu dividir por silêncio, divide por tempo
        if len(chunks) == 1:
            chunk_length_ms = chunk_length * 1000
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        transcriptions = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            # Salva chunk temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_chunk:
                chunk.export(tmp_chunk.name, format="wav")
                
                # Transcreve chunk
                if engine == "google":
                    text, _ = transcribe_with_google(recognizer, tmp_chunk.name, language)
                else:
                    text, _ = transcribe_with_sphinx(recognizer, tmp_chunk.name)
                
                if text and text != "Não foi possível entender o áudio":
                    transcriptions.append({
                        'chunk': i + 1,
                        'start_time': i * chunk_length,
                        'text': text.strip()
                    })
                
                # Limpa arquivo temporário
                os.unlink(tmp_chunk.name)
                
                # Atualiza progresso
                progress_bar.progress((i + 1) / len(chunks))
        
        return transcriptions, engine.title()
        
    except Exception as e:
        st.error(f"Erro na transcrição de áudio longo: {str(e)}")
        return [], engine.title()

def format_timestamp(seconds):
    """Formata timestamp em formato legível"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    st.title("🎤 Transcritor de Áudio Robusto")
    st.markdown("### Transcrição usando SpeechRecognition (Google + PocketSphinx)")
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Engine de reconhecimento
        engine = st.selectbox(
            "Engine de Reconhecimento:",
            ["google", "sphinx"],
            index=0,
            format_func=lambda x: {
                "google": "🌐 Google Speech (Online)",
                "sphinx": "💻 PocketSphinx (Offline)"
            }[x],
            help="Google é mais preciso mas requer internet. PocketSphinx funciona offline."
        )
        
        # Idioma (apenas para Google)
        if engine == "google":
            languages = {
                "pt-BR": "🇧🇷 Português (Brasil)",
                "pt-PT": "🇵🇹 Português (Portugal)", 
                "en-US": "🇺🇸 Inglês (EUA)",
                "es-ES": "🇪🇸 Espanhol",
                "fr-FR": "🇫🇷 Francês",
                "de-DE": "🇩🇪 Alemão",
                "it-IT": "🇮🇹 Italiano",
                "ja-JP": "🇯🇵 Japonês",
                "ko-KR": "🇰🇷 Coreano",
                "zh-CN": "🇨🇳 Chinês"
            }
            
            selected_language = st.selectbox(
                "Idioma do áudio:",
                list(languages.keys()),
                format_func=lambda x: languages[x]
            )
        else:
            selected_language = "en-US"
            st.info("💡 PocketSphinx suporta principalmente inglês")
        
        # Configurações para áudios longos
        st.subheader("📏 Áudios Longos")
        chunk_length = st.slider(
            "Tamanho dos chunks (segundos):",
            min_value=10,
            max_value=60,
            value=30,
            help="Divide áudios longos em partes menores"
        )
        
        # Opções de saída
        st.subheader("📄 Opções de Saída")
        show_chunks = st.checkbox("Mostrar chunks separados", value=True)
        show_timestamps = st.checkbox("Mostrar timestamps", value=True)
        
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
        
        # Informações do engine selecionado
        if engine == "google":
            st.info("🌐 **Google Speech Recognition**\n- Alta precisão\n- Requer internet\n- Suporte multilíngue")
        else:
            st.info("💻 **PocketSphinx**\n- Funciona offline\n- Boa para inglês\n- Sem limites de uso")
        
        if uploaded_file:
            if st.button("🚀 Iniciar Transcrição", type="primary"):
                # Inicializa recognizer
                recognizer = get_speech_recognizer()
                
                try:
                    # Converte áudio
                    with st.spinner("Processando áudio..."):
                        converted_path, duration = convert_audio_to_wav(uploaded_file)
                    
                    if converted_path is None:
                        st.error("Falha na conversão do áudio")
                        return
                    
                    st.info(f"Duração do áudio: {duration:.1f} segundos")
                    
                    # Decide estratégia baseada na duração
                    start_time = time.time()
                    
                    if duration > chunk_length:
                        st.info(f"Áudio longo detectado. Dividindo em chunks de {chunk_length}s...")
                        transcriptions, engine_used = transcribe_large_audio(
                            recognizer, converted_path, engine, selected_language, chunk_length
                        )
                        
                        # Processa resultados
                        if transcriptions:
                            full_text = " ".join([t['text'] for t in transcriptions])
                        else:
                            full_text = "Não foi possível transcrever o áudio"
                            
                    else:
                        st.info("Áudio curto. Transcrevendo diretamente...")
                        if engine == "google":
                            full_text, engine_used = transcribe_with_google(recognizer, converted_path, selected_language)
                        else:
                            full_text, engine_used = transcribe_with_sphinx(recognizer, converted_path)
                        
                        transcriptions = [{
                            'chunk': 1,
                            'start_time': 0,
                            'text': full_text
                        }] if full_text != "Não foi possível entender o áudio" else []
                    
                    end_time = time.time()
                    
                    # Limpa arquivo temporário
                    os.unlink(converted_path)
                    
                    # Exibe resultados
                    st.success(f"Transcrição concluída em {end_time - start_time:.2f} segundos")
                    st.info(f"Engine usado: {engine_used}")
                    
                except Exception as e:
                    st.error(f"Erro durante a transcrição: {str(e)}")
                    return
    
    # Área de resultados
    if uploaded_file and 'transcriptions' in locals() and transcriptions:
        st.markdown("---")
        st.subheader("📝 Resultado da Transcrição")
        
        # Texto completo
        with st.expander("📄 Texto Completo", expanded=True):
            full_text = " ".join([t['text'] for t in transcriptions])
            st.text_area("", value=full_text, height=200, key="full_text")
            
            # Botão de download
            st.download_button(
                label="📥 Download Texto",
                data=full_text,
                file_name=f"transcricao_{uploaded_file.name}.txt",
                mime="text/plain"
            )
        
        # Chunks com timestamps
        if show_chunks and len(transcriptions) > 1:
            with st.expander("📋 Transcrição por Chunks"):
                for t in transcriptions:
                    if show_timestamps:
                        start_time = format_timestamp(t['start_time'])
                        st.write(f"**Chunk {t['chunk']} [{start_time}]**: {t['text']}")
                    else:
                        st.write(f"**Chunk {t['chunk']}**: {t['text']}")
        
        # Estatísticas
        with st.expander("📊 Estatísticas"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Chunks", len(transcriptions))
            
            with col2:
                if 'duration' in locals():
                    st.metric("Duração", f"{duration:.1f}s")
            
            with col3:
                word_count = len(full_text.split())
                st.metric("Palavras", word_count)
    
    elif uploaded_file and 'transcriptions' in locals() and not transcriptions:
        st.warning("Não foi possível transcrever o áudio. Tente:")
        st.markdown("- Verificar a qualidade do áudio")
        st.markdown("- Usar um engine diferente")
        st.markdown("- Ajustar o tamanho dos chunks")

    # Informações sobre o sistema
    with st.sidebar:
        st.markdown("---")
        st.subheader("ℹ️ Informações")
        
        st.markdown("**Engines Disponíveis:**")
        st.markdown("- 🌐 **Google Speech**: Alta precisão, requer internet")
        st.markdown("- 💻 **PocketSphinx**: Offline, ideal para inglês")
        
        st.markdown("---")
        st.markdown("**Tecnologias:**")
        st.markdown("- 🎤 SpeechRecognition")
        st.markdown("- 🎵 PyDub para processamento")
        st.markdown("- 🌟 Streamlit para interface")
        
        # Dicas
        with st.expander("💡 Dicas de Uso"):
            st.markdown("""
            **Para melhor precisão:**
            - Use áudios com boa qualidade
            - Evite ruído de fundo
            - Fale claramente
            - Use Google Speech quando possível
            
            **Para áudios longos:**
            - Ajuste o tamanho dos chunks
            - Use chunks menores para fala rápida
            - Chunks maiores para fala pausada
            """)

if __name__ == "__main__":
    main()
