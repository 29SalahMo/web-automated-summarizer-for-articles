# Web-Based Article Summarizer

A powerful web application for automatically summarizing articles and documents using state-of-the-art AI models. Supports both English and Arabic text summarization.

## Features

- **Multi-language Support**: Summarize text in English and Arabic
- **Multiple AI Models**: Choose from BART, T5, and mT5 models
- **File Upload**: Support for PDF, DOCX, and TXT files
- **Customizable Summaries**: Adjust summary length (short, medium, long) and tone (formal, casual, tweet-style)
- **Semantic Analysis**: Get confidence scores and semantic similarity metrics
- **Summary History**: View and manage your summarization history
- **Modern UI**: Beautiful, responsive design with dark/light mode and theme customization

## Technologies Used

- **Backend**: Flask (Python)
- **AI Models**: 
  - Facebook BART (English)
  - Google T5 (English)
  - mT5 Multilingual XLSum (Arabic)
- **NLP Libraries**: Transformers, Sentence Transformers
- **File Processing**: PyPDF2, python-docx
- **Frontend**: HTML5, CSS3, JavaScript

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "web summarizer"
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
   ```bash
   venv\Scripts\activate
   ```
   - On Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Select Language**: Choose between English or Arabic
2. **Input Text**: Either paste your text directly or upload a file (PDF, DOCX, or TXT)
3. **Choose Settings**:
   - Summary Length: Short (~50 words), Medium (~100 words), or Long (~200 words)
   - Model: Select BART or T5 for English (mT5 is automatically used for Arabic)
   - Tone: Default, Formal, Casual, or Tweet-style
4. **Click Summarize**: Wait for the AI to process your text
5. **View Results**: See your summary along with confidence and semantic similarity scores

## API Endpoints

- `GET /` - Main application page
- `POST /summarize` - Summarize text or file
- `GET /health` - Health check endpoint
- `GET /test` - Test model availability

## Deployment

### Heroku

1. Create a `Procfile` (already included):
```
web: gunicorn app:app
```

2. Install gunicorn:
```bash
pip install gunicorn
```

3. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

### Railway

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Flask app
3. Set environment variables if needed

### Other Platforms

The application can be deployed on any platform that supports Python/Flask applications:
- Render
- Fly.io
- DigitalOcean App Platform
- AWS Elastic Beanstalk
- Google Cloud Run

## Configuration

The application uses the following models by default:
- **English**: `facebook/bart-large-cnn`, `t5-base`
- **Arabic**: `csebuetnlp/mT5_multilingual_XLSum`

Models are automatically downloaded on first use from Hugging Face.

## File Size Limits

- Maximum file size: 5MB
- Supported formats: PDF, DOCX, TXT

## Performance Notes

- First run may take longer as models are downloaded
- Large documents (>2000 words) are automatically chunked for processing
- Processing time depends on text length and selected model

## Troubleshooting

### Models not loading
- Ensure you have a stable internet connection for initial model download
- Check that you have sufficient disk space (models can be several GB)
- Verify all dependencies are installed correctly

### Memory issues
- Close other applications to free up RAM
- Consider using a smaller model or processing shorter texts

## License

This project is part of a graduation project. All rights reserved.

## Contributing

This is a graduation project. Contributions and suggestions are welcome!

## Contact

For questions or issues, please open an issue on the GitHub repository.
