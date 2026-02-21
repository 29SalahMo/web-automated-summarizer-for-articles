"""
Web-Based Automated Article Summarizer
Streamlit Application
"""

import os
import io
import re
import sys
import traceback

# Import Streamlit first for error display
try:
    import streamlit as st
except ImportError:
    print("ERROR: streamlit is not installed. Please install it with: pip install streamlit")
    sys.exit(1)

# Early error check - display a simple message if Streamlit context is not ready
try:
    # Test if Streamlit is ready
    _ = st.__version__
except:
    pass  # Streamlit not initialized yet, that's okay

# Import other dependencies with error handling
_imports_ok = True
_import_error = None
try:
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
    # Only define these if imports succeeded
    TOKENIZER_CLASSES = {
        "bart": BartTokenizer,
        "t5": T5Tokenizer,
        "pegasus": PegasusTokenizer,
    }
except ImportError as e:
    _imports_ok = False
    _import_error = str(e)
    # Define empty dict if imports fail
    TOKENIZER_CLASSES = {}


# -------------------------
# Model loading (same logic as Flask app, adapted)
# -------------------------

MODEL_OPTIONS = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-base",
    # Pegasus kept as option but may be heavy on free tiers
    # "pegasus": "google/pegasus-cnn_dailymail",
}

# TOKENIZER_CLASSES is defined in the import block above

@st.cache_resource(show_spinner=False)
def load_english_model(model_key):
    """Load a single English model on demand"""
    if not _imports_ok or model_key not in MODEL_OPTIONS:
        print(f"ERROR: Cannot load model '{model_key}': imports failed or model not in options")
        return None
    if model_key not in TOKENIZER_CLASSES:
        print(f"ERROR: Tokenizer class not found for '{model_key}'")
        return None
    try:
        model_name = MODEL_OPTIONS[model_key]
        print(f"Loading English model: {model_name}...")
        tokenizer_class = TOKENIZER_CLASSES[model_key]
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipeline_obj = pipeline("summarization", model=model, tokenizer=tokenizer)
        print(f"Successfully loaded English model: {model_name}")
        return pipeline_obj
    except Exception as e:
        error_msg = f"Failed to load English model '{model_key}': {str(e)}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        return None

@st.cache_resource(show_spinner=False)
def load_arabic_model():
    """Load Arabic model on demand"""
    if not _imports_ok:
        print("ERROR: Cannot load Arabic model: imports failed")
        return None
    try:
        arabic_model_name = "csebuetnlp/mT5_multilingual_XLSum"
        print(f"Loading Arabic model: {arabic_model_name}...")
        arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model_name)
        arabic_model = AutoModelForSeq2SeqLM.from_pretrained(arabic_model_name)
        print("Arabic model and tokenizer loaded, creating pipeline...")
        # Try pipeline first
        try:
            arabic_pipeline = pipeline("summarization", model=arabic_model, tokenizer=arabic_tokenizer)
            print("Successfully created Arabic pipeline")
            return {"pipeline": arabic_pipeline, "type": "pipeline"}
        except Exception as pipeline_error:
            # If pipeline fails, store model and tokenizer for manual generation
            print(f"Pipeline creation failed, using manual generation: {str(pipeline_error)[:200]}")
            return {
                "model": arabic_model,
                "tokenizer": arabic_tokenizer,
                "type": "manual"
            }
    except Exception as e:
        error_msg = f"Failed to load Arabic model: {str(e)}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        return None


@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load sentence transformer for semantic similarity"""
    if not _imports_ok:
        return None
    try:
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        print(f"Warning: Failed to load embedder: {e}")
        return None


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

    # Pick summarizer (already loaded in main function)
        if language == "english":
            summarizer = summarizers.get(model_choice)
            if not summarizer:
                # Try any available English model
                for key, val in summarizers.items():
                    if key != "arabic" and val is not None:
                        summarizer = val
                        model_choice = key
                        break
                if not summarizer:
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
                if language == "arabic":
                    arabic_data = summarizers.get("arabic")
                    if not arabic_data:
                        raise RuntimeError("Arabic model not loaded. Please try again or check the logs.")
                    
                    if arabic_data.get("type") == "pipeline":
                        # Use pipeline if available
                        try:
                            arabic_pipeline = arabic_data["pipeline"]
                            result = arabic_pipeline(chunk, max_length=max_len, min_length=min_len, do_sample=False)
                            summary_parts.append(result[0]["summary_text"])
                        except Exception as pipe_error:
                            raise RuntimeError(f"Pipeline error: {str(pipe_error)[:200]}")
                    elif arabic_data.get("type") == "manual":
                        # Manual generation for mT5
                        try:
                            arabic_model = arabic_data["model"]
                            arabic_tokenizer = arabic_data["tokenizer"]
                            
                            if arabic_model is None or arabic_tokenizer is None:
                                raise RuntimeError("Arabic model or tokenizer is None")
                            
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
                            if summary_text:
                                summary_parts.append(summary_text)
                            else:
                                raise RuntimeError("Generated empty summary")
                        except Exception as gen_error:
                            raise RuntimeError(f"Arabic generation error: {str(gen_error)[:200]}")
                    else:
                        raise RuntimeError("Invalid Arabic model configuration.")
                else:
                    # English models use pipeline
                    summarizer = summarizers.get(model_choice)
                    if not summarizer:
                        # Try any available English model
                        for key, val in summarizers.items():
                            if key != "arabic" and val is not None:
                                summarizer = val
                                model_choice = key
                                break
                        if not summarizer:
                            raise RuntimeError("No English models available.")
                    
                    try:
                        result = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
                        summary_parts.append(result[0]["summary_text"])
                    except Exception as sum_error:
                        raise RuntimeError(f"Summarization error: {str(sum_error)[:200]}")
            except Exception as e:
                error_msg = f"Error processing chunk {i+1}/{len(chunks)}: {str(e)}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                st.error(f"❌ {error_msg}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                raise RuntimeError(error_msg)
        
        summary = " ".join(summary_parts)

        if length == "short":
            sentences = re.split(r"(?<=[.!?]) +", summary)
            summary = " ".join(sentences[:2]).strip()

        confidence = round((1 - len(summary) / len(original_text)) * 100, 2)
    
    # Calculate semantic similarity if embedder is available
    semantic_score = 0.0
    if embedder is not None:
        try:
            embeddings = embedder.encode([original_text, summary], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            semantic_score = round(similarity * 100, 2)
        except Exception as e:
            print(f"Warning: Could not calculate semantic similarity: {e}")
            semantic_score = 0.0

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
    """Main Streamlit application"""
    # Check imports first
    if not _imports_ok:
        st.error(f"❌ Missing required package: {_import_error}")
        st.info("Please install all requirements: pip install -r requirements.txt")
        return
    
    # Set page config first (must be first Streamlit command)
    try:
        st.set_page_config(
            page_title="Web-Based Article Summarizer",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass  # Already set
    
    # Add custom CSS for professional neon-themed UI with excellent visibility
    st.markdown("""
    <style>
    /* Main app background - professional dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    
    /* Make content readable with semi-transparent backgrounds */
    .main .block-container {
        background: rgba(15, 15, 35, 0.95);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(108, 99, 255, 0.4);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    }
    
    /* Style sidebar - FIXED TEXT VISIBILITY */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.95) !important;
        border-right: 2px solid rgba(108, 99, 255, 0.4);
    }
    
    /* Sidebar text - make all text visible */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar markdown text */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    /* Sidebar labels - make them very visible */
    [data-testid="stSidebar"] label {
        color: #e8e8e8 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    
    /* Sidebar select boxes - ensure text is visible with high contrast */
    [data-testid="stSidebar"] .stSelectbox > div > div > select {
        background: #1a1a2e !important;
        border: 2px solid rgba(108, 99, 255, 0.8) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 1rem !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div > select:focus {
        background: #252540 !important;
        border-color: #6c63ff !important;
        box-shadow: 0 0 25px rgba(108, 99, 255, 0.9) !important;
        color: #ffffff !important;
        outline: none !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div > select:hover {
        background: #252540 !important;
        border-color: rgba(108, 99, 255, 1) !important;
        color: #ffffff !important;
    }
    
    /* Sidebar selectbox options - high contrast */
    [data-testid="stSidebar"] .stSelectbox option {
        background: #1a1a2e !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        font-size: 1.05rem !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox option:hover,
    [data-testid="stSidebar"] .stSelectbox option:focus,
    [data-testid="stSidebar"] .stSelectbox option:checked {
        background: #6c63ff !important;
        color: #ffffff !important;
    }
    
    /* Force sidebar select text color */
    [data-testid="stSidebar"] select {
        color: #ffffff !important;
        background-color: #1a1a2e !important;
    }
    
    [data-testid="stSidebar"] select option {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
    }
    
    /* Sidebar info boxes */
    [data-testid="stSidebar"] .stInfo {
        background: rgba(33, 150, 243, 0.2) !important;
        border-left: 4px solid #2196f3 !important;
        color: #ffffff !important;
        border-radius: 5px;
        padding: 0.75rem !important;
    }
    
    [data-testid="stSidebar"] .stInfo * {
        color: #ffffff !important;
    }
    
    /* Sidebar caption text */
    [data-testid="stSidebar"] .stCaption {
        color: #d0d0d0 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar metric containers */
    [data-testid="stSidebar"] [data-testid="stMetricContainer"] {
        background: rgba(20, 20, 40, 0.7) !important;
        border: 1px solid rgba(108, 99, 255, 0.4) !important;
        border-radius: 10px;
        padding: 1rem !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #a084e8 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #c8c8c8 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #6c63ff 0%, #8b7fff 100%);
        color: #ffffff !important;
        border: 2px solid rgba(108, 99, 255, 0.6);
        font-weight: 600 !important;
    }
    
    /* Sidebar markdown content */
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] .stMarkdown b {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar code blocks */
    [data-testid="stSidebar"] code {
        background: rgba(20, 20, 40, 0.8) !important;
        color: #00ff88 !important;
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    
    /* Style headers and text for better visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Style buttons with neon glow effect */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff 0%, #8b7fff 100%);
        color: #ffffff !important;
        border: 2px solid rgba(108, 99, 255, 0.6);
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.5), 0 0 15px rgba(108, 99, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7d73ff 0%, #9b8fff 100%);
        box-shadow: 0 6px 25px rgba(108, 99, 255, 0.7), 0 0 20px rgba(108, 99, 255, 0.5);
        transform: translateY(-2px);
        border-color: rgba(108, 99, 255, 0.8);
    }
    
    /* Style select boxes - FIXED TEXT VISIBILITY WITH HIGH CONTRAST */
    .stSelectbox > div > div > select {
        background: #1a1a2e !important;
        border: 2px solid rgba(108, 99, 255, 0.7) !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.6rem 1rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
    }
    
    .stSelectbox > div > div > select:focus {
        background: #252540 !important;
        border-color: #6c63ff !important;
        box-shadow: 0 0 20px rgba(108, 99, 255, 0.8) !important;
        outline: none !important;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div > select:hover {
        background: #252540 !important;
        border-color: rgba(108, 99, 255, 0.9) !important;
        color: #ffffff !important;
    }
    
    /* Style selectbox options dropdown - ensure visibility */
    .stSelectbox option {
        background: #1a1a2e !important;
        color: #ffffff !important;
        padding: 0.8rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox option:hover,
    .stSelectbox option:focus,
    .stSelectbox option:checked {
        background: #6c63ff !important;
        color: #ffffff !important;
    }
    
    /* Force text color for select elements */
    select {
        color: #ffffff !important;
        background-color: #1a1a2e !important;
    }
    
    select option {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
    }
    
    /* Style text areas - FIXED TEXT VISIBILITY */
    .stTextArea > div > div > textarea {
        background: rgba(20, 20, 40, 0.95) !important;
        border: 2px solid rgba(108, 99, 255, 0.5) !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #6c63ff !important;
        box-shadow: 0 0 15px rgba(108, 99, 255, 0.6) !important;
        outline: none;
        background: rgba(25, 25, 45, 0.95) !important;
    }
    
    /* Style file uploader */
    .stFileUploader > div {
        background: rgba(20, 20, 40, 0.95) !important;
        border: 2px dashed rgba(108, 99, 255, 0.5) !important;
        border-radius: 10px;
        padding: 1.5rem;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(108, 99, 255, 0.8) !important;
        background: rgba(25, 25, 45, 0.95) !important;
    }
    
    /* Style labels - make them visible */
    label {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Make metrics stand out with neon colors */
    [data-testid="stMetricValue"] {
        color: #a084e8 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b8b8b8 !important;
        font-weight: 500 !important;
    }
    
    /* Style success/error messages with neon borders */
    .stSuccess {
        background: rgba(76, 175, 80, 0.15) !important;
        border-left: 4px solid #4caf50 !important;
        color: #ffffff !important;
        border-radius: 5px;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.15) !important;
        border-left: 4px solid #f44336 !important;
        color: #ffffff !important;
        border-radius: 5px;
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.15) !important;
        border-left: 4px solid #2196f3 !important;
        color: #ffffff !important;
        border-radius: 5px;
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.15) !important;
        border-left: 4px solid #ff9800 !important;
        color: #ffffff !important;
        border-radius: 5px;
    }
    
    /* Add glow to main title */
    .stTitle {
        text-align: center;
    }
    
    .stTitle h1 {
        background: linear-gradient(135deg, #6c63ff 0%, #a084e8 50%, #ff1493 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 0 0 30px rgba(108, 99, 255, 0.3);
    }
    
    /* Ensure all text is readable */
    .stMarkdown, p, span, div {
        color: #e0e0e0 !important;
    }
    
    /* Style expander headers */
    .streamlit-expanderHeader {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(20, 20, 40, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(108, 99, 255, 0.3) !important;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(108, 99, 255, 0.2) !important;
        border-bottom: 2px solid #6c63ff !important;
        color: #ffffff !important;
    }
    
    /* Style download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #ffffff !important;
        border: 2px solid rgba(0, 212, 255, 0.6);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1ae0ff 0%, #00aadd 100%);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.6);
    }
    
    /* Style caption text */
    .stCaption {
        color: #b8b8b8 !important;
    }
    
    /* Style code blocks */
    code {
        background: rgba(20, 20, 40, 0.8) !important;
        color: #00ff88 !important;
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 4px;
        padding: 0.2rem 0.4rem;
    }
    
    /* Improve spacing and readability */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Style metric containers */
    [data-testid="stMetricContainer"] {
        background: rgba(20, 20, 40, 0.6);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌐 Web-Based Automated Article Summarizer")
    
    # Add interactive header with stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✨ AI Models", "3", help="BART, T5, and mT5 models available")
    with col2:
        st.metric("🌍 Languages", "2", help="English and Arabic support")
    with col3:
        st.metric("📄 Formats", "3", help="PDF, DOCX, and TXT files")
    with col4:
        if 'summary_count' not in st.session_state:
            st.session_state.summary_count = 0
        st.metric("📊 Summaries", st.session_state.summary_count, help="Total summaries created")
    
    st.markdown(
        "Summarize long articles in **English** and **Arabic** using state-of-the-art transformer models."
    )
    
    # Add interactive tips section
    with st.expander("💡 Quick Tips & Examples", expanded=False):
        tip_col1, tip_col2 = st.columns(2)
        with tip_col1:
            st.info("""
            **💡 Pro Tips:**
            - Use **Medium** length for balanced summaries
            - **BART** works best for news articles
            - **T5** is faster for general text
            - Upload PDFs for research papers
            """)
        with tip_col2:
            if st.button("📝 Load Sample Text", use_container_width=True):
                sample_text = """Artificial intelligence has revolutionized many aspects of our daily lives, from the way we communicate to how we work and learn. Machine learning algorithms can now process vast amounts of data, recognize patterns, and make predictions with remarkable accuracy. Natural language processing enables computers to understand and generate human language, powering virtual assistants and translation services. Computer vision allows machines to interpret and analyze visual information, driving advances in autonomous vehicles and medical imaging. As AI technology continues to evolve, it promises to solve complex problems and create new opportunities across various industries."""
                st.session_state.sample_text = sample_text
                st.success("Sample text loaded! Check the text area below.")
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.sample_text = ""
                st.rerun()

    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Add model info with tooltips
        st.markdown("### 🤖 AI Model Selection")
        language = st.selectbox("Language", ["english", "arabic"], index=0, help="Choose the language of your text")

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

    tab_text, tab_file, tab_examples = st.tabs(["✍️ Paste Text", "📄 Upload File", "📚 Examples"])

    input_text = ""
    uploaded_file = None

    with tab_text:
        # Check for sample text
        if 'sample_text' in st.session_state and st.session_state.sample_text:
            input_text = st.text_area(
                "Paste your article here",
                value=st.session_state.sample_text,
                height=250,
                placeholder="Paste an article, report, or any long text...",
                key="text_input",
                help="Paste your text here or use the sample text button above"
            )
            st.session_state.sample_text = ""  # Clear after use
        else:
            input_text = st.text_area(
                "Paste your article here",
                height=250,
                placeholder="Paste an article, report, or any long text...",
                key="text_input",
                help="Paste your text here or use the sample text button above"
            )
        
        # Real-time word count with visual indicator
        if input_text.strip():
            word_count = len(input_text.strip().split())
            char_count = len(input_text.strip())
            
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                st.metric("📝 Words", word_count)
            with col_w2:
                st.metric("🔤 Characters", f"{char_count:,}")
            with col_w3:
                # Estimate reading time
                reading_time = max(1, word_count // 200)  # Average 200 words per minute
                st.metric("⏱️ Reading Time", f"~{reading_time} min")
            
            # Progress bar for text length
            if word_count > 0:
                progress_pct = min(100, (word_count / 5000) * 100)  # Scale to 5000 words
                st.progress(progress_pct / 100)
                if word_count < 100:
                    st.warning("⚠️ Very short text. Summary options may be limited.")
                elif word_count > 5000:
                    st.info("ℹ️ Large text will be processed in chunks automatically.")

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

            # Create progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.info("🔄 Initializing AI models...")
                progress_bar.progress(10)
                
                try:
                    # Load models on demand
                    summarizers = {}
                    embedder = None
                    
                    if language == "english":
                        status_text.info(f"🤖 Loading {model_choice.upper()} model...")
                        progress_bar.progress(30)
                        # Load only the selected English model
                        summarizer = load_english_model(model_choice)
                        if summarizer:
                            summarizers[model_choice] = summarizer
                            progress_bar.progress(50)
                            status_text.success(f"✅ {model_choice.upper()} model loaded successfully!")
                        else:
                            # Try fallback models
                            status_text.warning(f"⚠️ {model_choice.upper()} not available, trying alternatives...")
                            for fallback_key in MODEL_OPTIONS.keys():
                                if fallback_key != model_choice:
                                    fallback = load_english_model(fallback_key)
                                    if fallback:
                                        summarizers[fallback_key] = fallback
                                        model_choice = fallback_key
                                        progress_bar.progress(50)
                                        status_text.success(f"✅ Using {fallback_key.upper()} model instead")
                                        break
                    else:
                        # Load Arabic model
                        status_text.info("🌍 Loading Arabic (mT5) model...")
                        progress_bar.progress(30)
                        arabic_data = load_arabic_model()
                        if arabic_data:
                            summarizers["arabic"] = arabic_data
                            progress_bar.progress(50)
                            status_text.success("✅ Arabic model loaded successfully!")
                        else:
                            status_text.error("❌ Failed to load Arabic model")
                    
                    if not summarizers:
                        progress_bar.progress(0)
                        status_text.error("❌ Failed to load any models")
                        st.error("""
                        **Model Loading Failed**
                        
                        Possible reasons:
                        - Network connection issues (models need to be downloaded)
                        - Insufficient memory/resources
                        - Model download interrupted
                        
                        **Solutions:**
                        1. Check your internet connection
                        2. Wait a moment and try again (first-time download can take 2-5 minutes)
                        3. Try refreshing the page
                        4. For Arabic: The mT5 model is large and may need more time/resources
                        """)
                        with st.expander("🔍 Technical Details"):
                            st.code("Check the browser console or Streamlit logs for detailed error messages")
                        return
                    
                    status_text.info("🧠 Loading semantic analysis model...")
                    progress_bar.progress(70)
                    embedder = load_embedder()
                    if embedder:
                        progress_bar.progress(80)
                        status_text.success("✅ All models loaded! Processing your text...")
                        progress_bar.progress(90)
                    else:
                        status_text.warning("⚠️ Semantic analysis model not available, continuing without it...")
                        progress_bar.progress(90)
                        
                except Exception as model_error:
                    progress_bar.progress(0)
                    status_text.error("❌ Model loading error")
                    error_details = str(model_error)
                    st.error(f"**Model Loading Error:** {error_details}")
                    st.info("""
                    **This might be due to:**
                    - Memory limitations (models are large)
                    - Network issues (downloading models)
                    - Resource constraints on the server
                    
                    **Please try:**
                    1. Wait a moment and try again
                    2. Try with a different model
                    3. Refresh the page
                    """)
                    with st.expander("🔍 Full Error Details"):
                        st.code(traceback.format_exc())
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

            # Action buttons
            st.markdown("---")
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                st.download_button(
                    "💾 Download TXT",
                    data=result["summary"],
                    file_name=f"summary_{st.session_state.summary_count}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with action_col2:
                if st.button("🔄 Summarize Again", use_container_width=True):
                    st.rerun()
            with action_col3:
                if st.button("📊 View Stats", use_container_width=True):
                    st.info(f"""
                    **Session Statistics:**
                    - Total Summaries: {st.session_state.summary_count}
                    - Total Words Processed: {st.session_state.total_words_processed:,}
                    - Average Words per Summary: {st.session_state.total_words_processed // max(1, st.session_state.summary_count):,}
                    """)

        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
            st.code(traceback.format_exc())
            st.info("💡 Tip: Try with a shorter text or check if models are loading correctly.")


# Streamlit executes the entire file
# Wrap main() in try-except to catch and display any errors
if __name__ == "__main__" or True:  # Always execute for Streamlit
    try:
        main()
    except Exception as e:
        # Display error in Streamlit UI
        try:
            st.error(f"❌ **Application Error:** {str(e)}")
            with st.expander("🔍 Click to see full error details"):
                st.code(traceback.format_exc())
            st.warning("💡 If this error persists, check the Hugging Face Spaces logs.")
        except Exception as e2:
            # If we can't display error, at least try to show something
            try:
                st.error("❌ An error occurred. Please check the logs.")
            except:
                # Last resort - print to console
                import sys
                print(f"FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
