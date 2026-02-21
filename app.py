import os
import io
import re
import sys
import traceback

import streamlit as st
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    PegasusTokenizer,
    BartTokenizer,
    T5Tokenizer,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import torch


# -------------------------
# Model loading (same logic as Flask app, adapted)
# -------------------------

MODEL_OPTIONS = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-base",
    # Pegasus kept as option but may be heavy on free tiers
    # "pegasus": "google/pegasus-cnn_dailymail",
}

TOKENIZER_CLASSES = {
    "bart": BartTokenizer,
    "t5": T5Tokenizer,
    "pegasus": PegasusTokenizer,
}

@st.cache_resource(show_spinner=True)
def load_summarizers():
    summarizers = {}
    english_models_loaded = 0

    for key, model_name in MODEL_OPTIONS.items():
        try:
            tokenizer_class = TOKENIZER_CLASSES[key]
            tokenizer = tokenizer_class.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            summarizers[key] = pipeline("summarization", model=model, tokenizer=tokenizer)
            english_models_loaded += 1
        except Exception as e:
            st.warning(f"Failed to load English model '{key}': {e}")
            summarizers[key] = None

    # Arabic model - mT5 needs special handling
    arabic_model_loaded = False
    try:
        arabic_model_name = "csebuetnlp/mT5_multilingual_XLSum"
        with st.spinner("Loading Arabic model (this may take a while)..."):
            arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model_name)
            arabic_model = AutoModelForSeq2SeqLM.from_pretrained(arabic_model_name)
            # Store both pipeline and raw model/tokenizer for flexibility
            try:
                # Try pipeline first
                arabic_pipeline = pipeline("summarization", model=arabic_model, tokenizer=arabic_tokenizer)
                summarizers["arabic"] = {"pipeline": arabic_pipeline, "type": "pipeline"}
            except Exception as pipe_error:
                # If pipeline fails, store model and tokenizer for manual generation
                st.info("ℹ️ Using manual generation for Arabic model (pipeline not available)")
                summarizers["arabic"] = {
                    "model": arabic_model,
                    "tokenizer": arabic_tokenizer,
                    "type": "manual"
                }
        arabic_model_loaded = True
    except Exception as e:
        error_msg = str(e)
        st.warning(f"⚠️ Failed to load Arabic model: {error_msg[:200]}")
        summarizers["arabic"] = None

    return summarizers, english_models_loaded, arabic_model_loaded


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def chunk_text(text: str, max_chunk: int = 1000, language: str = "english"):
    if language == "arabic":
        sentences = re.split(r"(?<=[.!؟]) +", text)
    elif language == "english":
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            nltk.download("punkt", quiet=True)
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r"(?<=[.!?]) +", text)
    else:
        sentences = re.split(r"(?<=[.!?]) +", text)

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_chunk:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks


def summarize_text(
    summarizers,
    embedder,
    original_text: str,
    model_choice: str,
    language: str,
    tone: str,
    length: str,
):
    # Get Arabic data if needed
    arabic_data = None
    if language == "arabic":
        arabic_data = summarizers.get("arabic")
    length_map = {
        "short": (50, 10),
        "medium": (120, 40),
        "long": (250, 100),
    }
    max_len, min_len = length_map.get(length, (130, 30))

    word_count = len(original_text.strip().split())
    short_input_warning = None
    if word_count < 100:
        short_input_warning = "Summary length options may not affect very short texts."

    if language == "english":
        if tone == "formal":
            original_text = f"Please summarize this formally:\n{original_text}"
        elif tone == "casual":
            original_text = f"Summarize casually:\n{original_text}"
        elif tone == "tweet":
            original_text = f"Summarize in a tweet style:\n{original_text}"

    # Pick summarizer
    if language == "english":
        summarizer = summarizers.get(model_choice)
        if not summarizer:
            available_english = [
                k for k, v in summarizers.items() if v is not None and k != "arabic"
            ]
            if available_english:
                summarizer = summarizers[available_english[0]]
                model_choice = available_english[0]
            else:
                raise RuntimeError("No English models available.")
    else:
        arabic_data = summarizers.get("arabic")
        if not arabic_data:
            raise RuntimeError("Arabic model not available. Please check if the model loaded successfully.")
        # Arabic will be handled in the processing loop
        summarizer = None

    # Chunking
    chunk_size = 2000 if word_count > 2000 else 1000
    chunks = chunk_text(original_text, max_chunk=chunk_size, language=language)
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        raise RuntimeError("Could not create valid chunks from input text.")

    summary_parts = []
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
            try:
                if language == "arabic" and arabic_data:
                    if arabic_data.get("type") == "pipeline":
                        # Use pipeline if available
                        arabic_pipeline = arabic_data["pipeline"]
                        result = arabic_pipeline(chunk, max_length=max_len, min_length=min_len, do_sample=False)
                        summary_parts.append(result[0]["summary_text"])
                    elif arabic_data.get("type") == "manual":
                        # Manual generation for mT5
                        arabic_model = arabic_data["model"]
                        arabic_tokenizer = arabic_data["tokenizer"]
                        
                        # Add task prefix for mT5 (XLSum format)
                        input_text_mt5 = f"summarize: {chunk}"
                        inputs = arabic_tokenizer(
                            input_text_mt5, 
                            return_tensors="pt", 
                            max_length=512, 
                            truncation=True,
                            padding=True
                        )
                        
                        # Move to same device as model
                        device = next(arabic_model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Generate summary
                        with torch.no_grad():
                            outputs = arabic_model.generate(
                                **inputs,
                                max_length=max_len,
                                min_length=min_len,
                                num_beams=4,
                                early_stopping=True,
                                no_repeat_ngram_size=3
                            )
                        
                        summary_text = arabic_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        summary_parts.append(summary_text)
                    else:
                        raise RuntimeError("Arabic model configuration error.")
                else:
                    # English models use pipeline
                    if not summarizer:
                        raise RuntimeError("Summarizer not available.")
                    result = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
                    summary_parts.append(result[0]["summary_text"])
            except Exception as e:
                error_msg = f"Error processing chunk {i+1}/{len(chunks)}: {str(e)}"
                st.error(error_msg)
                st.code(traceback.format_exc())
                raise RuntimeError(error_msg)

    summary = " ".join(summary_parts)

    if length == "short":
        sentences = re.split(r"(?<=[.!?]) +", summary)
        summary = " ".join(sentences[:2]).strip()

    confidence = round((1 - len(summary) / len(original_text)) * 100, 2)
    embeddings = embedder.encode([original_text, summary], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    semantic_score = round(similarity * 100, 2)

    return {
        "summary": summary,
        "confidence": confidence,
        "semantic_similarity": semantic_score,
        "warning": short_input_warning,
        "model_used": "mT5" if language == "arabic" else model_choice,
    }


# -------------------------
# Streamlit UI
# -------------------------

def main():
    try:
        st.set_page_config(
            page_title="Web-Based Article Summarizer",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass  # Already set

    st.title("🌐 Web-Based Automated Article Summarizer")
    st.markdown(
        "Summarize long articles in **English** and **Arabic** using state-of-the-art transformer models."
    )

    with st.sidebar:
        st.header("Settings")
        language = st.selectbox("Language", ["english", "arabic"], index=0)

        if language == "english":
            model_choice = st.selectbox("Model", ["bart", "t5"], index=0)
        else:
            model_choice = "arabic"
            st.info("Arabic uses the mT5 multilingual XL-Sum model.")

        tone = st.selectbox(
            "Tone (English only)",
            ["default", "formal", "casual", "tweet"],
            index=0,
        )
        length = st.selectbox(
            "Summary length",
            ["short", "medium", "long"],
            index=1,
        )

        st.markdown("---")
        st.caption(
            "Note: First run may take a while to download models. Subsequent runs are faster."
        )

    tab_text, tab_file = st.tabs(["✍️ Paste Text", "📄 Upload File"])

    input_text = ""
    uploaded_file = None

    with tab_text:
        input_text = st.text_area(
            "Paste your article here",
            height=250,
            placeholder="Paste an article, report, or any long text...",
        )
        if input_text.strip():
            st.caption(f"Word count: {len(input_text.strip().split())}")

    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload a file (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"]
        )

    if st.button("Summarize", type="primary"):
        if not input_text.strip() and not uploaded_file:
            st.error("Please paste some text or upload a file.")
            return

        try:
            if not input_text.strip() and uploaded_file:
                file_bytes = uploaded_file.read()
                name = uploaded_file.name.lower()
                if name.endswith(".pdf"):
                    input_text = extract_text_from_pdf(file_bytes)
                elif name.endswith(".docx"):
                    input_text = extract_text_from_docx(file_bytes)
                else:  # .txt
                    input_text = file_bytes.decode("utf-8", errors="ignore")

            if not input_text.strip():
                st.error("The provided text/file appears to be empty.")
                return

            with st.spinner("Loading models (if not already loaded)..."):
                try:
                    summarizers, english_loaded, arabic_loaded = load_summarizers()
                    embedder = load_embedder()
                except Exception as model_error:
                    st.error(f"Failed to load models: {str(model_error)}")
                    st.info("This might be due to memory limitations or network issues. Please try again.")
                    return

            result = summarize_text(
                summarizers=summarizers,
                embedder=embedder,
                original_text=input_text,
                model_choice=model_choice,
                language=language,
                tone=tone,
                length=length,
            )

            st.subheader("Summary")
            st.write(result["summary"])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['confidence']}%")
            with col2:
                st.metric("Semantic Similarity", f"{result['semantic_similarity']}%")
            with col3:
                st.metric(
                    "Model Used",
                    result["model_used"].upper() if result["model_used"] else "-",
                )

            if result["warning"]:
                st.info(result["warning"])

            st.download_button(
                "Download Summary as TXT",
                data=result["summary"],
                file_name="summary.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
            st.code(traceback.format_exc())
            st.info("💡 Tip: Try with a shorter text or check if models are loading correctly.")


# Streamlit runs the entire file, so call main() directly
try:
    main()
except Exception as e:
    st.error(f"❌ Application Error: {str(e)}")
    with st.expander("🔍 Show Full Error Details"):
        st.code(traceback.format_exc())
    st.info("💡 Please check the logs or try refreshing the page. If the error persists, check the Hugging Face Spaces logs.")
