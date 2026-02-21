from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoModelForSeq2SeqLM, PegasusTokenizer, BartTokenizer, T5Tokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import os
import docx
import io
import re
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
except ImportError:
    sent_tokenize = None

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# English model options
# Note: pegasus is excluded due to stability issues (causes crashes on some systems)
# The system works perfectly with bart and t5 models
MODEL_OPTIONS = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-base",
    # "pegasus": "google/pegasus-cnn_dailymail"  # Disabled due to crash issues
}

# Tokenizers
TOKENIZER_CLASSES = {
    "bart": BartTokenizer,
    "t5": T5Tokenizer,
    "pegasus": PegasusTokenizer
}

# Load English models
summarizers = {}
english_models_loaded = 0
for key, model_name in MODEL_OPTIONS.items():
    try:
        print(f"Loading {key} model: {model_name}")
        tokenizer_class = TOKENIZER_CLASSES[key]
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizers[key] = pipeline("summarization", model=model, tokenizer=tokenizer)
        print(f"✓ {key} model loaded successfully")
        english_models_loaded += 1
    except Exception as e:
        print(f"✗ Failed to load {key} model: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        summarizers[key] = None

# Load Arabic model
arabic_model_loaded = False
try:
    print("Loading Arabic model: csebuetnlp/mT5_multilingual_XLSum")
    arabic_model_name = "csebuetnlp/mT5_multilingual_XLSum"
    arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model_name)
    arabic_model = AutoModelForSeq2SeqLM.from_pretrained(arabic_model_name)
    summarizers["arabic"] = pipeline("summarization", model=arabic_model, tokenizer=arabic_tokenizer)
    print("✓ Arabic model loaded successfully")
    arabic_model_loaded = True
except Exception as e:
    print(f"✗ Failed to load Arabic model: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    summarizers["arabic"] = None

print(f"Available models: {list(summarizers.keys())}")
print(f"Working models: {[k for k, v in summarizers.items() if v is not None]}")
print(f"English models loaded: {english_models_loaded}/2 (bart, t5)")
print(f"Arabic model loaded: {arabic_model_loaded}")

# Check if we have any working models
working_models = [k for k, v in summarizers.items() if v is not None]
if not working_models:
    print("❌ CRITICAL ERROR: No models loaded successfully!")
    print("Please check your internet connection and try again.")
    print("Make sure you have installed all required dependencies:")
    print("pip install transformers torch sentence-transformers PyPDF2 python-docx nltk sentencepiece protobuf")
else:
    print(f"✅ {len(working_models)} model(s) ready for use")
    print(f"Working models: {working_models}")

# Embedder for semantic similarity
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def extract_text_from_pdf(file_storage):
    reader = PyPDF2.PdfReader(io.BytesIO(file_storage.read()))
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file_storage):
    file_storage.seek(0)
    doc = docx.Document(io.BytesIO(file_storage.read()))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def chunk_text(text, max_chunk=1000, language='english'):
    # Use sentence tokenization for better chunking
    if language == 'arabic':
        # Simple regex-based Arabic sentence splitting
        sentences = re.split(r'(?<=[.!؟]) +', text)
    elif language == 'english':
        # Try NLTK first, fallback to regex if it fails
        try:
            if sent_tokenize:
                sentences = sent_tokenize(text)
            else:
                raise ImportError("NLTK not available")
        except Exception as e:
            print(f"NLTK sentence tokenization failed, using regex fallback: {str(e)}")
            # Fallback: split by period, exclamation, question mark
            sentences = re.split(r'(?<=[.!?]) +', text)
    else:
        # Fallback: split by period
        sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current = ''
    for sent in sentences:
        if len(current) + len(sent) <= max_chunk:
            current += (' ' if current else '') + sent
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    
    return chunks

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health_check():
    """Simple health check endpoint"""
    working_models = [k for k, v in summarizers.items() if v is not None]
    return jsonify({
        "status": "healthy",
        "working_models": working_models,
        "total_models": len(summarizers)
    })

@app.route("/test")
def test_models():
    """Test endpoint to check model status"""
    working_models = [k for k, v in summarizers.items() if v is not None]
    english_models = [k for k, v in summarizers.items() if v is not None and k != "arabic"]
    arabic_models = [k for k, v in summarizers.items() if v is not None and k == "arabic"]
    
    return jsonify({
        "all_models": list(summarizers.keys()),
        "working_models": working_models,
        "english_models": english_models,
        "arabic_models": arabic_models,
        "english_models_loaded": len(english_models),
        "arabic_models_loaded": len(arabic_models),
        "total_working": len(working_models)
    })

@app.route("/summarize", methods=["POST"])
def summarize():
    print("=== SUMMARIZE REQUEST RECEIVED ===")
    original_text = ""
    model_choice = request.form.get("model", "bart")
    language = request.form.get("language", "english")
    tone = request.form.get("tone", "default")
    length = request.form.get("length", "medium")
    
    print(f"Model: {model_choice}, Language: {language}, Tone: {tone}, Length: {length}")

    length_map = {
        "short": (50, 10),
        "medium": (120, 40),
        "long": (250, 100)
    }
    max_len, min_len = length_map.get(length, (130, 30))

    # File size and type validation
    allowed_extensions = {"pdf", "docx", "txt"}
    max_file_size = 5 * 1024 * 1024  # 5MB

    if request.form.get("text"):
        original_text = request.form["text"]
        print(f"Text input received, length: {len(original_text)}")
        if not original_text.strip():
            print("ERROR: Empty text input")
            return jsonify({"error": "Please enter some text to summarize."})
    elif "file" in request.files:
        file = request.files["file"]
        filename = file.filename.lower()
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        print(f"File upload: {filename}, extension: {ext}")
        if ext not in allowed_extensions:
            print(f"ERROR: Unsupported file type: {ext}")
            return jsonify({"error": "Unsupported file type. Only PDF, DOCX, and TXT files are allowed."})
        file.seek(0, 2)  # Move to end of file
        file_length = file.tell()
        file.seek(0)  # Reset pointer
        if file_length > max_file_size:
            print(f"ERROR: File too large: {file_length} bytes")
            return jsonify({"error": "File too large. Maximum allowed size is 5MB."})
        try:
            if ext == "pdf":
                original_text = extract_text_from_pdf(file)
            elif ext == "docx":
                original_text = extract_text_from_docx(file)
            elif ext == "txt":
                original_text = file.read().decode("utf-8")
            print(f"Extracted text length: {len(original_text)}")
        except Exception as e:
            print(f"ERROR extracting text: {str(e)}")
            return jsonify({"error": f"Error reading file: {str(e)}"})
        if not original_text.strip():
            print("ERROR: Empty file content")
            return jsonify({"error": "The uploaded file appears to be empty or could not be read."})
    else:
        print("ERROR: No input provided")
        return jsonify({"error": "No input provided. Please enter text or upload a file."})

    # Warn if input is very short
    word_count = len(original_text.strip().split())
    print(f"Word count: {word_count}")
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

    try:
        print("Getting summarizer...")
        if language == "english":
            summarizer = summarizers.get(model_choice)
            if not summarizer:
                print(f"ERROR: English model {model_choice} not available")
                print(f"Available models: {list(summarizers.keys())}")
                print(f"Working models: {[k for k, v in summarizers.items() if v is not None]}")
                # Try to find any available English model as fallback
                available_english_models = [k for k, v in summarizers.items() if v is not None and k != "arabic"]
                if available_english_models:
                    fallback_model = available_english_models[0]
                    print(f"Using fallback English model: {fallback_model}")
                    summarizer = summarizers[fallback_model]
                else:
                    print("ERROR: No English models available")
                    print(f"Available models: {list(summarizers.keys())}")
                    print(f"Working models: {[k for k, v in summarizers.items() if v is not None]}")
                    return jsonify({
                        "error": f"English summarization model '{model_choice}' is not available. Please try Arabic language or check server logs.",
                        "available_models": [k for k, v in summarizers.items() if v is not None],
                        "requested_model": model_choice,
                        "working_models": [k for k, v in summarizers.items() if v is not None]
                    })
        else:
            summarizer = summarizers.get("arabic")
            if not summarizer:
                print("ERROR: Arabic model not available")
                return jsonify({"error": "Arabic model not available"})
        
        print(f"Summarizer found: {type(summarizer)}")
        if summarizer is None:
            print("ERROR: Summarizer is None despite being found")
            return jsonify({"error": "Model initialization error. Please try again."})
        
        # Dynamically adjust chunk size for large articles
        if word_count > 2000:
            chunk_size = 2000
        else:
            chunk_size = 1000
        print(f"Chunking with size: {chunk_size}")
        chunks = chunk_text(original_text, max_chunk=chunk_size, language=language)
        print(f"Created {len(chunks)} chunks")
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        if not chunks:
            print("ERROR: No valid chunks created")
            return jsonify({"error": "The input could not be processed into valid text chunks for summarization."})
        
        print("Starting summarization...")
        summary_parts = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                result = summarizer(chunk, max_length=max_len, min_length=min_len)
                chunk_summary = result[0]["summary_text"]
                summary_parts.append(chunk_summary)
                print(f"Chunk {i+1} summary length: {len(chunk_summary)}")
            except Exception as e:
                print(f"ERROR processing chunk {i+1}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                return jsonify({"error": f"Error during summarization: {str(e)}"})
        
        summary = " ".join(summary_parts)
        print(f"Final summary length: {len(summary)}")

        # For 'short' summaries, truncate to first 1-2 sentences
        if length == "short":
            import re
            sentences = re.split(r'(?<=[.!?]) +', summary)
            summary = " ".join(sentences[:2]).strip()

        confidence = round((1 - len(summary) / len(original_text)) * 100, 2)
        embeddings = embedder.encode([original_text, summary], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        semantic_score = round(similarity * 100, 2)

        response = {
            "summary": summary,
            "confidence": f"{confidence}%",
            "semantic_similarity": f"{semantic_score}%",
            "model_used": "mT5" if language == "arabic" else model_choice
        }
        if short_input_warning:
            response["warning"] = short_input_warning
        print("=== SUMMARIZATION COMPLETED SUCCESSFULLY ===")
        return jsonify(response)

    except Exception as e:
        print(f"ERROR in summarization: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during summarization: {str(e)}"})

if __name__ == "__main__":
    # use_reloader=False prevents Flask from restarting, which causes issues
    # when loading large ML models at startup
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
