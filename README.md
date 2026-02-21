---
title: Web-Based Automated Article Summarizer
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# Web-Based Automated Article Summarizer

A powerful web application for automatically summarizing articles and documents using state-of-the-art AI models. Supports both English and Arabic text summarization.

## Features

- **Multi-language Support**: Summarize text in English and Arabic
- **Multiple AI Models**: Choose from BART, T5, and mT5 models
- **File Upload**: Support for PDF, DOCX, and TXT files
- **Customizable Summaries**: Adjust summary length (short, medium, long) and tone (formal, casual, tweet-style)
- **Semantic Analysis**: Get confidence scores and semantic similarity metrics
- **Modern UI**: Beautiful, responsive design with Streamlit

## Technologies Used

- **Backend**: Streamlit (Python)
- **AI Models**: 
  - Facebook BART (English)
  - Google T5 (English)
  - mT5 Multilingual XLSum (Arabic)
- **NLP Libraries**: Transformers, Sentence Transformers
- **File Processing**: PyPDF2, python-docx

## Usage

1. **Select Language**: Choose between English or Arabic
2. **Input Text**: Either paste your text directly or upload a file (PDF, DOCX, or TXT)
3. **Choose Settings**:
   - Summary Length: Short (~50 words), Medium (~100 words), or Long (~200 words)
   - Model: Select BART or T5 for English (mT5 is automatically used for Arabic)
   - Tone: Default, Formal, Casual, or Tweet-style (English only)
4. **Click Summarize**: Wait for the AI to process your text
5. **View Results**: See your summary along with confidence and semantic similarity scores

## File Size Limits

- Maximum file size: 5MB
- Supported formats: PDF, DOCX, TXT

## Performance Notes

- First run may take longer as models are downloaded
- Large documents (>2000 words) are automatically chunked for processing
- Processing time depends on text length and selected model

## License

This project is part of a graduation project. All rights reserved.
