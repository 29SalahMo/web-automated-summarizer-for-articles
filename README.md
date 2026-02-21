
# 🌐 Web-Based Automated Article Summarizer

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge)](https://huggingface.co/transformers)

A powerful, production-ready web application for automatically summarizing articles and documents using state-of-the-art AI transformer models. Supports both **English** and **Arabic** text summarization with customizable length, tone, and multiple AI models.

## ✨ Features

- 🌍 **Multi-language Support**: Summarize text in English and Arabic
- 🤖 **Multiple AI Models**: 
  - **BART** (Facebook) - Best for general English summarization
  - **T5** (Google) - Versatile English summarization
  - **mT5** (Multilingual) - Arabic and multilingual support
- 📄 **File Upload Support**: PDF, DOCX, and TXT files
- 🎨 **Customizable Summaries**: 
  - Summary length: Short (~50 words), Medium (~100 words), Long (~200 words)
  - Tone options: Default, Formal, Casual, Tweet-style (English only)
- 📊 **Semantic Analysis**: Confidence scores and semantic similarity metrics
- 🚀 **Fast & Efficient**: Lazy model loading - models load only when needed
- 💻 **Modern UI**: Beautiful, responsive Streamlit interface

## 🚀 Quick Start

### Option 1: Use Online (Recommended)

**Deployed on Hugging Face Spaces**: [Try it now!](https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME)

Simply visit the link above and start summarizing!

### Option 2: Run Locally

#### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- 4GB+ RAM (for model loading)
- Internet connection (for downloading models on first run)

#### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/29SalahMo/web-automated-summarizer-for-articles.git
   cd web-automated-summarizer-for-articles
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

#### Windows Quick Start (Batch File)

If you're on Windows, simply double-click `run_summarizer.bat` - it will:
- Check for virtual environment
- Activate it automatically
- Start the Streamlit app
- Open in your browser

## 📖 How to Use

### Web Interface

1. **Select Language**: Choose between English or Arabic from the sidebar
2. **Choose Model** (English only): Select BART or T5
3. **Set Summary Length**: Short, Medium, or Long
4. **Choose Tone** (English only): Default, Formal, Casual, or Tweet-style
5. **Input Text**: 
   - **Option A**: Paste your text directly in the "Paste Text" tab
   - **Option B**: Upload a file (PDF, DOCX, or TXT) in the "Upload File" tab
6. **Click "Summarize"**: Wait for processing (first run may take 2-3 minutes to download models)
7. **View Results**: 
   - Read your summary
   - Check confidence and semantic similarity scores
   - Download summary as TXT file

### Command Line / Batch Processing

#### Python Script Example

Create a file `batch_summarize.py`:

```python
import sys
from app import load_english_model, load_arabic_model, load_embedder, summarize_text

# Load models once
print("Loading models...")
english_model = load_english_model("bart")  # or "t5"
arabic_model = load_arabic_model()
embedder = load_embedder()

# Summarize multiple texts
texts = [
    "Your first long article text here...",
    "Your second article text here...",
    # Add more texts
]

summarizers = {"bart": english_model, "arabic": arabic_model}

for i, text in enumerate(texts):
    print(f"\nProcessing text {i+1}...")
    result = summarize_text(
        summarizers=summarizers,
        embedder=embedder,
        original_text=text,
        model_choice="bart",
        language="english",
        tone="default",
        length="medium"
    )
    print(f"Summary: {result['summary']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Semantic Similarity: {result['semantic_similarity']}%")
```

Run it:
```bash
python batch_summarize.py
```

#### Batch File Processing

Create a script to process multiple files:

```python
import os
from app import load_english_model, load_embedder, summarize_text, extract_text_from_pdf, extract_text_from_docx

# Load model once
model = load_english_model("bart")
embedder = load_embedder()
summarizers = {"bart": model}

# Process all files in a directory
input_dir = "input_files"
output_dir = "summaries"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.pdf', '.docx', '.txt')):
        print(f"Processing {filename}...")
        
        # Extract text
        filepath = os.path.join(input_dir, filename)
        if filename.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                text = extract_text_from_pdf(f.read())
        elif filename.endswith('.docx'):
            with open(filepath, 'rb') as f:
                text = extract_text_from_docx(f.read())
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Summarize
        result = summarize_text(
            summarizers=summarizers,
            embedder=embedder,
            original_text=text,
            model_choice="bart",
            language="english",
            tone="default",
            length="medium"
        )
        
        # Save summary
        output_path = os.path.join(output_dir, f"{filename}_summary.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Original: {filename}\n")
            f.write(f"Summary Length: {len(result['summary'])} words\n")
            f.write(f"Confidence: {result['confidence']}%\n")
            f.write(f"Semantic Similarity: {result['semantic_similarity']}%\n\n")
            f.write("SUMMARY:\n")
            f.write(result['summary'])
        
        print(f"✓ Saved summary to {output_path}")
```

## 🎯 Use Cases

- **Academic Research**: Quickly summarize research papers and articles
- **News Articles**: Get quick summaries of long news pieces
- **Business Reports**: Extract key points from lengthy business documents
- **Content Creation**: Generate summaries for blog posts and articles
- **Language Learning**: Summarize texts in different languages
- **Document Analysis**: Process multiple documents in batch

## 🔧 Configuration

### Model Selection

- **BART** (`facebook/bart-large-cnn`): Best for general-purpose English summarization
- **T5** (`t5-base`): Good balance of speed and quality
- **mT5** (`csebuetnlp/mT5_multilingual_XLSum`): For Arabic and multilingual text

### Summary Lengths

- **Short**: ~50 words (2-3 sentences)
- **Medium**: ~100 words (4-6 sentences) - Recommended
- **Long**: ~200 words (8-10 sentences)

### Supported File Formats

- **PDF**: `.pdf` files (text extraction)
- **Word Documents**: `.docx` files
- **Text Files**: `.txt` files (UTF-8 encoding)

**File Size Limits**: 
- Maximum file size: 5MB (for web interface)
- No limit for command-line processing

## 📊 Performance

- **First Run**: 2-5 minutes (model download)
- **Subsequent Runs**: 10-30 seconds (model loading from cache)
- **Processing Time**: 
  - Short texts (<500 words): 5-10 seconds
  - Medium texts (500-2000 words): 15-30 seconds
  - Long texts (>2000 words): 30-60 seconds (chunked processing)

## 🛠️ Technical Details

### Architecture

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python 3.11+
- **AI Models**: Hugging Face Transformers
- **NLP Libraries**: 
  - Transformers (Hugging Face)
  - Sentence Transformers (for semantic similarity)
  - NLTK (for sentence tokenization)

### Dependencies

Key packages:
- `streamlit>=1.28.0` - Web framework
- `transformers>=4.30.0` - AI models
- `sentence-transformers>=2.2.0` - Semantic analysis
- `torch>=2.0.0` - Deep learning framework
- `PyPDF2>=3.0.0` - PDF processing
- `python-docx>=0.8.11` - Word document processing

See `requirements.txt` for complete list.

### Model Information

| Model | Language | Size | Best For |
|-------|----------|------|----------|
| BART Large CNN | English | ~1.6GB | General articles, news |
| T5 Base | English | ~850MB | Balanced performance |
| mT5 XLSum | Arabic/Multilingual | ~2.3GB | Arabic articles, multilingual |

## 🐛 Troubleshooting

### Common Issues

**1. "Oh no. Error running app"**
- **Solution**: Check that all dependencies are installed: `pip install -r requirements.txt`
- Restart the app: Press `Ctrl+C` and run `streamlit run app.py` again

**2. Models not loading**
- **Solution**: Check internet connection (models download on first run)
- Wait 2-5 minutes for initial model download
- Check available disk space (models need ~5GB)

**3. Memory errors**
- **Solution**: Close other applications
- Use smaller models (T5 instead of BART)
- Process shorter texts

**4. Arabic summarization fails**
- **Solution**: Ensure Arabic model downloaded successfully
- Check that text contains Arabic characters
- Try with shorter Arabic text first

### Getting Help

- **Check Logs**: Look at terminal/console output for detailed error messages
- **GitHub Issues**: Report bugs at [GitHub Issues](https://github.com/29SalahMo/web-automated-summarizer-for-articles/issues)
- **Model Issues**: Check [Hugging Face Model Cards](https://huggingface.co/models)

## 📝 License

This project is part of a graduation project. All rights reserved.

## 🙏 Acknowledgments

- **Hugging Face** for providing transformer models
- **Streamlit** for the amazing web framework
- **Facebook AI** for BART model
- **Google** for T5 model
- **CSE BUET NLP** for mT5 Arabic model

## 📧 Contact

For questions, suggestions, or collaboration:
- **GitHub**: [29SalahMo](https://github.com/29SalahMo)
- **Repository**: [web-automated-summarizer-for-articles](https://github.com/29SalahMo/web-automated-summarizer-for-articles)

## 🚀 Deployment

### Hugging Face Spaces

This app is configured for easy deployment on Hugging Face Spaces:

1. Fork this repository
2. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
3. Create a new Space
4. Select "Streamlit" as SDK
5. Connect your GitHub repository
6. Deploy!

The `README.md` file contains the necessary metadata for Hugging Face Spaces.

### Other Platforms

The app can also be deployed on:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Use `Procfile` and `runtime.txt`
- **Railway**: Supports Python apps
- **Render**: Free tier available

## 📈 Future Enhancements

- [ ] Support for more languages
- [ ] Batch processing UI
- [ ] API endpoint for programmatic access
- [ ] Export summaries in multiple formats (PDF, DOCX)
- [ ] Custom model fine-tuning
- [ ] Real-time collaboration features

---

**Made with ❤️ for efficient text summarization**