# 📊 Model Testing Tables for PowerPoint Presentation

## Table 1: Model Performance Overview

| Model | Success Rate | Avg Time (s) | Avg Similarity | Avg Compression |
|-------|-------------|--------------|----------------|-----------------|
| BART  | 100.0%      | 2.345        | 85.2%          | 12.3%           |
| T5    | 100.0%      | 1.892        | 82.7%          | 15.1%           |
| PEGASUS | 100.0%   | 3.156        | 87.1%          | 10.8%           |
| Arabic | 100.0%     | 2.789        | 83.5%          | 13.2%           |

## Table 2: Detailed Performance by Text Length

| Text Length | Model | Success | Time (s) | Similarity | Compression | Words |
|-------------|-------|---------|----------|------------|-------------|-------|
| Short       | BART  | ✓       | 1.234    | 87.2%      | 15.3%       | 45    |
| Short       | T5    | ✓       | 0.987    | 84.1%      | 18.2%       | 52    |
| Short       | PEGASUS | ✓   | 1.567    | 89.3%      | 12.1%       | 38    |
| Short       | Arabic | ✓     | 1.345    | 85.7%      | 14.8%       | 42    |
| Medium      | BART  | ✓       | 2.456    | 85.1%      | 11.8%       | 89    |
| Medium      | T5    | ✓       | 1.987    | 82.3%      | 14.2%       | 107   |
| Medium      | PEGASUS | ✓   | 3.234    | 86.9%      | 10.1%       | 76    |
| Medium      | Arabic | ✓     | 2.678    | 83.2%      | 12.9%       | 93    |
| Long        | BART  | ✓       | 3.345    | 83.3%      | 9.8%        | 156   |
| Long        | T5    | ✓       | 2.801    | 81.7%      | 12.8%       | 204   |
| Long        | PEGASUS | ✓   | 4.567    | 85.1%      | 8.2%        | 130   |
| Long        | Arabic | ✓     | 3.945    | 81.6%      | 11.7%       | 178   |

## Table 3: Model Rankings

| Category | Best Model | Score | Details |
|----------|------------|-------|---------|
| 🏃 Speed | T5 | 1.892s | Fastest processing time |
| 🎯 Accuracy | PEGASUS | 87.1% | Highest semantic similarity |
| 📝 Compression | PEGASUS | 10.8% | Best text compression |
| 🏆 Overall | BART | 0.234 | Best balanced performance |

## Table 4: Model Specifications

| Model | Architecture | Parameters | Training Data | Specialization |
|-------|-------------|------------|---------------|----------------|
| BART | Encoder-Decoder | 400M | CNN/DailyMail | News summarization |
| T5 | Encoder-Decoder | 220M | C4 corpus | General text tasks |
| PEGASUS | Encoder-Decoder | 568M | C4 + HugeNews | Abstractive summarization |
| Arabic (mT5) | Encoder-Decoder | 580M | Multilingual | Arabic text processing |

## Table 5: Testing Methodology

| Aspect | Description | Metric |
|--------|-------------|--------|
| **Text Lengths** | Short (150 words), Medium (300 words), Long (800 words) | Character count |
| **Processing Time** | Time from input to summary generation | Seconds |
| **Semantic Similarity** | How well summary preserves original meaning | Percentage (0-100%) |
| **Compression Ratio** | How much text is condensed | Percentage of original |
| **Success Rate** | Percentage of successful summarizations | Percentage (0-100%) |
| **Word Count** | Number of words in generated summary | Integer |

## Table 6: Key Findings

| Finding | Model | Value | Significance |
|---------|-------|-------|--------------|
| Fastest Processing | T5 | 1.892s | Best for real-time applications |
| Most Accurate | PEGASUS | 87.1% | Best for quality-critical tasks |
| Best Compression | PEGASUS | 10.8% | Most concise summaries |
| Most Reliable | All | 100% | All models work consistently |
| Best for English | BART | Balanced | Good all-around performance |
| Best for Arabic | mT5 | 83.5% | Specialized for Arabic text |

## Table 7: Recommendations by Use Case

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| **Real-time Applications** | T5 | Fastest processing |
| **High-Quality Summaries** | PEGASUS | Highest accuracy |
| **Arabic Content** | mT5 | Specialized for Arabic |
| **General Purpose** | BART | Balanced performance |
| **News Articles** | BART | Trained on news data |
| **Academic Papers** | PEGASUS | Better for complex texts |

---

## 📋 How to Use These Tables:

1. **Copy the table format** into your PowerPoint
2. **Replace the sample data** with your actual test results
3. **Use consistent formatting** (colors, fonts, borders)
4. **Add charts/graphs** based on the numerical data
5. **Include screenshots** of your web application

## 🎯 Testing Script Usage:

Run the testing script to get real data:
```bash
python test_models_simple.py
```

This will generate actual performance metrics for your models! 