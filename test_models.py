#!/usr/bin/env python3
"""
Test script to check model loading and diagnose issues
"""

import sys
import traceback

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print(f"✓ Sentence Transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"✗ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print(f"✓ PyPDF2 {PyPDF2.__version__}")
    except ImportError as e:
        print(f"✗ PyPDF2 import failed: {e}")
        return False
    
    try:
        import docx
        print(f"✓ python-docx")
    except ImportError as e:
        print(f"✗ python-docx import failed: {e}")
        return False
    
    try:
        import nltk
        print(f"✓ NLTK {nltk.__version__}")
    except ImportError as e:
        print(f"✗ NLTK import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from transformers import pipeline, AutoModelForSeq2SeqLM, BartTokenizer, T5Tokenizer, PegasusTokenizer, AutoTokenizer
        
        # Test English models
        english_models = {
            "bart": "facebook/bart-large-cnn",
            "t5": "t5-base",
            "pegasus": "google/pegasus-cnn_dailymail"
        }
        
        tokenizer_classes = {
            "bart": BartTokenizer,
            "t5": T5Tokenizer,
            "pegasus": PegasusTokenizer
        }
        
        loaded_models = {}
        
        for key, model_name in english_models.items():
            try:
                print(f"Loading {key} model: {model_name}")
                tokenizer_class = tokenizer_classes[key]
                tokenizer = tokenizer_class.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
                loaded_models[key] = summarizer
                print(f"✓ {key} model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {key} model: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                traceback.print_exc()
                loaded_models[key] = None
        
        # Test Arabic model
        try:
            print("Loading Arabic model: csebuetnlp/mT5_multilingual_XLSum")
            arabic_model_name = "csebuetnlp/mT5_multilingual_XLSum"
            arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model_name)
            arabic_model = AutoModelForSeq2SeqLM.from_pretrained(arabic_model_name)
            arabic_summarizer = pipeline("summarization", model=arabic_model, tokenizer=arabic_tokenizer)
            loaded_models["arabic"] = arabic_summarizer
            print("✓ Arabic model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Arabic model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            loaded_models["arabic"] = None
        
        # Summary
        working_models = [k for k, v in loaded_models.items() if v is not None]
        english_working = [k for k, v in loaded_models.items() if v is not None and k != "arabic"]
        
        print(f"\nSummary:")
        print(f"Total models: {len(loaded_models)}")
        print(f"Working models: {working_models}")
        print(f"English models working: {english_working}")
        print(f"Arabic model working: {'arabic' in working_models}")
        
        if not english_working:
            print("\n❌ CRITICAL: No English models are working!")
            print("This explains why English summarization is not working.")
            print("Please check your internet connection and try again.")
        
        return len(english_working) > 0
        
    except Exception as e:
        print(f"✗ Model loading test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("=== Model Loading Diagnostic Tool ===\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies:")
        print("pip install flask transformers torch sentence-transformers PyPDF2 python-docx nltk sentencepiece protobuf")
        return
    
    # Test model loading
    english_models_work = test_model_loading()
    
    if english_models_work:
        print("\n✅ English models are working! The issue might be elsewhere.")
    else:
        print("\n❌ English models are not working. This is the root cause.")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Make sure you have enough disk space")
        print("3. Try running: pip install --upgrade transformers torch")
        print("4. Check if you have any firewall/proxy blocking downloads")

if __name__ == "__main__":
    main() 