import streamlit as st
import whisper
import tempfile
import os
from pathlib import Path
import time
from datetime import datetime
import io

# Configuração da página
st.set_page_config(
    page_title="Transcritor de Áudio",
    page_icon="🎤",
    layout="wide"
)

# Cache do modelo para evitar recarregamento
@st.cache_resource
def load_whisper_model(model_name):
    """Carrega o modelo Whisper com cache"""
    try:
        model = whisper.load_model(model_name)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

def transcribe_audio(audio_file, model, language=None):
    """Realiza a transcrição do áudio"""
    try:
        # Salva o arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Configura opções de transcrição
        options = {"fp16": False}
        if language and language != "auto":
            options["language"] = language
        
        # Realiza a transcrição
        result = model.transcribe(tmp_file_path, **options)
        
        # Remove arquivo temporário
        os.unlink(tmp_file_path)
        
        return result
    
    except Exception as e:
        # Limpa arquivo temporário em caso de erro
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise e

def format_timestamp(seconds):
    """Formata timestamp em formato legível"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    st.title("🎤 Transcritor de Áudio com Whisper")
    st.markdown("---")
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Seleção do modelo
        model_options = {
            "tiny": "Tiny (39 MB) - Mais rápido, menor precisão",
            "base": "Base (74 MB) - Equilibrado",
            "small": "Small (244 MB) - Boa precisão",
            "medium": "Medium (769 MB) - Alta precisão",
            "large": "Large (1550 MB) - Máxima precisão"
        }
        
        selected_model = st.selectbox(
            "Modelo Whisper:",
            options=list(model_options.keys()),
            index=1,  # Default: base
            format_func=lambda x: model_options[x]
        )
        
        # Seleção do idioma
        language_options = {
            "auto": "Detecção automática",
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
            "Idioma:",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x]
        )
        
        st.markdown("---")
        st.markdown("### 📝 Instruções")
        st.markdown("""
        1. Selecione o modelo desejado
        2. Escolha o idioma (ou deixe automático)
        3. Faça upload do arquivo de áudio
        4. Clique em 'Transcrever'
        5. Baixe o resultado se desejar
        """)
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload do Áudio")
        
        # Upload do arquivo
        uploaded_file = st.file_uploader(
            "Escolha um arquivo de áudio",
            type=['wav', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'webm', 'flac'],
            help="Formatos suportados: WAV, MP3, MP4, MPEG, MPGA, M4A, WebM, FLAC"
        )
        
        if uploaded_file is not None:
            # Mostra informações do arquivo
            file_details = {
                "Nome": uploaded_file.name,
                "Tamanho": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "Tipo": uploaded_file.type
            }
            
            st.success("Arquivo carregado com sucesso!")
            
            # Exibe detalhes do arquivo
            with st.expander("📊 Detalhes do Arquivo"):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            
            # Player de áudio
            st.audio(uploaded_file, format=uploaded_file.type)
    
    with col2:
        st.header("🔧 Status")
        
        # Informações sobre o modelo selecionado
        st.info(f"**Modelo:** {model_options[selected_model]}")
        st.info(f"**Idioma:** {language_options[selected_language]}")
        
        # Botão de transcrição
        if uploaded_file is not None:
            if st.button("🚀 Iniciar Transcrição", type="primary", use_container_width=True):
                with st.spinner("Carregando modelo..."):
                    model = load_whisper_model(selected_model)
                
                if model is not None:
                    try:
                        with st.spinner("Transcrevendo áudio... Isso pode levar alguns minutos."):
                            start_time = time.time()
                            
                            # Progresso
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            
                            # Simula progresso (Whisper não fornece callback real)
                            for i in range(100):
                                time.sleep(0.1)
                                progress_bar.progress(i + 1)
                                progress_text.text(f"Processando... {i+1}%")
                            
                            # Realiza transcrição
                            result = transcribe_audio(
                                uploaded_file, 
                                model, 
                                selected_language if selected_language != "auto" else None
                            )
                            
                            end_time = time.time()
                            processing_time = end_time - start_time
                            
                            progress_bar.empty()
                            progress_text.empty()
                            
                            st.success(f"✅ Transcrição concluída em {processing_time:.2f} segundos!")
                            
                            # Armazena resultado na sessão
                            st.session_state.transcription_result = result
                            st.session_state.processing_time = processing_time
                    
                    except Exception as e:
                        st.error(f"❌ Erro durante a transcrição: {str(e)}")
        else:
            st.warning("⚠️ Faça upload de um arquivo de áudio primeiro")
    
    # Exibe resultados se disponíveis
    if hasattr(st.session_state, 'transcription_result'):
        st.markdown("---")
        st.header("📄 Resultado da Transcrição")
        
        result = st.session_state.transcription_result
        
        # Tabs para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Texto Completo", "⏱️ Segmentos", "📊 Estatísticas", "💾 Download"])
        
        with tab1:
            st.subheader("Texto Transcrito")
            transcribed_text = result["text"].strip()
            st.text_area(
                "Transcrição:",
                value=transcribed_text,
                height=300,
                help="Você pode copiar este texto"
            )
        
        with tab2:
            st.subheader("Transcrição com Timestamps")
            
            for i, segment in enumerate(result["segments"]):
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                with st.container():
                    col_time, col_text = st.columns([1, 4])
                    with col_time:
                        st.code(f"{start_time} - {end_time}")
                    with col_text:
                        st.write(text)
                    
                    if i < len(result["segments"]) - 1:
                        st.divider()
        
        with tab3:
            st.subheader("Estatísticas da Transcrição")
            
            # Calcula estatísticas
            total_duration = result["segments"][-1]["end"] if result["segments"] else 0
            word_count = len(transcribed_text.split())
            char_count = len(transcribed_text)
            segment_count = len(result["segments"])
            
            # Exibe em colunas
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("⏱️ Duração", f"{total_duration:.1f}s")
            
            with stat_col2:
                st.metric("📝 Palavras", word_count)
            
            with stat_col3:
                st.metric("🔤 Caracteres", char_count)
            
            with stat_col4:
                st.metric("📑 Segmentos", segment_count)
            
            # Informações adicionais
            st.markdown("### Informações do Processamento")
            processing_info = f"""
            - **Tempo de processamento:** {st.session_state.processing_time:.2f} segundos
            - **Modelo utilizado:** {selected_model}
            - **Idioma detectado:** {result.get('language', 'N/A')}
            - **Velocidade:** {word_count / st.session_state.processing_time:.1f} palavras/segundo
            """
            st.markdown(processing_info)
        
        with tab4:
            st.subheader("Download dos Resultados")
            
            # Prepare diferentes formatos
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Texto simples
            txt_content = transcribed_text
            st.download_button(
                label="📄 Baixar como TXT",
                data=txt_content,
                file_name=f"transcricao_{timestamp}.txt",
                mime="text/plain"
            )
            
            # Texto com timestamps
            timestamped_content = ""
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                timestamped_content += f"[{start_time} - {end_time}] {text}\n\n"
            
            st.download_button(
                label="⏱️ Baixar com Timestamps",
                data=timestamped_content,
                file_name=f"transcricao_timestamps_{timestamp}.txt",
                mime="text/plain"
            )
            
            # SRT (legenda)
            srt_content = ""
            for i, segment in enumerate(result["segments"], 1):
                start_srt = f"{int(segment['start']//3600):02d}:{int((segment['start']%3600)//60):02d}:{int(segment['start']%60):02d},000"
                end_srt = f"{int(segment['end']//3600):02d}:{int((segment['end']%3600)//60):02d}:{int(segment['end']%60):02d},000"
                srt_content += f"{i}\n{start_srt} --> {end_srt}\n{segment['text'].strip()}\n\n"
            
            st.download_button(
                label="🎬 Baixar como SRT (Legenda)",
                data=srt_content,
                file_name=f"transcricao_{timestamp}.srt",
                mime="text/plain"
            )

if __name__ == "__main__":
    # Inicialização
    if "transcription_result" not in st.session_state:
        st.session_state.transcription_result = None
    
    main()