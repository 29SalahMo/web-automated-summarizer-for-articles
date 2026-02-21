#!/usr/bin/env python3
"""
Simple Model Testing Script for Web Summarizer
"""

import time
import json
from datetime import datetime

# Test texts of different lengths
test_texts = {
    "short": "Artificial intelligence is transforming the way we live and work. Machine learning algorithms can now process vast amounts of data to make predictions and automate complex tasks. This technology is being applied across industries from healthcare to finance.",
    
    "medium": """The COVID-19 pandemic has fundamentally changed the global landscape in unprecedented ways. When the virus first emerged in late 2019, few could have predicted the scale of its impact. Governments worldwide implemented lockdowns, travel restrictions, and social distancing measures to curb the spread. The healthcare system faced immense pressure as hospitals overflowed with patients. Economic activity ground to a halt, leading to widespread job losses and business closures. Remote work became the new normal for many industries, accelerating the adoption of digital technologies. Education systems had to rapidly adapt to online learning platforms. The pandemic also highlighted existing inequalities in healthcare access and economic stability. While vaccines have provided hope for recovery, the long-term effects on society, economy, and mental health continue to unfold. The experience has taught valuable lessons about global cooperation, scientific communication, and the importance of resilient healthcare systems.""",
    
    "long": """The rapid advancement of artificial intelligence and machine learning technologies is fundamentally transforming various sectors of society, from healthcare and education to transportation and entertainment. These technologies, which enable computers to learn from data and make decisions with minimal human intervention, are becoming increasingly sophisticated and accessible. In healthcare, AI systems can analyze medical images to detect diseases like cancer at earlier stages than traditional methods, potentially saving countless lives. Machine learning algorithms can predict patient outcomes, optimize treatment plans, and even assist in drug discovery by analyzing vast datasets of molecular structures and clinical trials. The education sector is also experiencing significant changes, with AI-powered tutoring systems providing personalized learning experiences tailored to individual student needs and learning styles. These systems can adapt their teaching methods based on student performance, offering additional support where needed and accelerating progress in areas where students excel. Transportation is being revolutionized by autonomous vehicles, which use complex AI systems to navigate roads, interpret traffic signals, and make split-second decisions to ensure passenger safety. While fully autonomous vehicles are still in development, many modern cars already incorporate AI features like adaptive cruise control and lane-keeping assistance. The entertainment industry has embraced AI for content creation, recommendation systems, and even scriptwriting assistance. Streaming platforms use sophisticated algorithms to suggest content based on viewing history and preferences, while AI tools help creators generate music, art, and written content. However, the widespread adoption of AI also raises important ethical and societal questions. Concerns about job displacement as automation replaces human workers in various industries are growing, requiring careful consideration of retraining programs and social safety nets. Privacy issues arise as AI systems collect and analyze vast amounts of personal data, necessitating robust regulations and ethical guidelines. Algorithmic bias, where AI systems perpetuate or amplify existing social inequalities, is another critical concern that must be addressed through diverse training data and inclusive design practices. The development of AI also raises questions about accountability and transparency, particularly in high-stakes applications like criminal justice and healthcare. As these technologies continue to evolve, it is essential to balance innovation with responsible development, ensuring that AI benefits society as a whole while minimizing potential harms. This requires collaboration between technologists, policymakers, ethicists, and the public to establish appropriate guidelines and regulations."""
}

def test_model_performance():
    """Test model performance and generate tables"""
    print("🤖 Web Summarizer Model Testing")
    print("=" * 50)
    
    # Import your app's models
    from app import summarizers, embedder
    
    results = {}
    
    # Test each model on different text lengths
    for length_name, text in test_texts.items():
        print(f"\nTesting {length_name.upper()} text ({len(text)} characters)...")
        results[length_name] = {}
        
        for model_name, summarizer in summarizers.items():
            if summarizer is None:
                continue
                
            print(f"  Testing {model_name}...")
            
            try:
                start_time = time.time()
                result = summarizer(text, max_length=150, min_length=50)
                summary = result[0]["summary_text"]
                processing_time = time.time() - start_time
                
                # Calculate metrics
                compression_ratio = len(summary) / len(text)
                word_count_original = len(text.split())
                word_count_summary = len(summary.split())
                
                # Calculate semantic similarity
                embeddings = embedder.encode([text, summary], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                
                results[length_name][model_name] = {
                    "summary": summary,
                    "processing_time": processing_time,
                    "compression_ratio": compression_ratio,
                    "word_count_original": word_count_original,
                    "word_count_summary": word_count_summary,
                    "semantic_similarity": similarity,
                    "success": True
                }
                
                print(f"    ✓ Success: {processing_time:.3f}s, {similarity*100:.1f}% similarity")
                
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
                results[length_name][model_name] = {
                    "error": str(e),
                    "success": False
                }
    
    return results

def generate_tables(results):
    """Generate comparison tables"""
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLES")
    print("="*60)
    
    # Table 1: Overall Performance
    print("\n📊 TABLE 1: OVERALL PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"{'Model':<12} {'Success Rate':<15} {'Avg Time (s)':<15} {'Avg Similarity':<15} {'Avg Compression':<15}")
    print("-" * 80)
    
    model_stats = {}
    for model_name in ["bart", "t5", "pegasus", "arabic"]:
        success_count = 0
        total_count = 0
        times = []
        similarities = []
        compressions = []
        
        for length_name, model_results in results.items():
            if model_name in model_results:
                result = model_results[model_name]
                total_count += 1
                if result and result.get("success"):
                    success_count += 1
                    times.append(result["processing_time"])
                    similarities.append(result["semantic_similarity"])
                    compressions.append(result["compression_ratio"])
        
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            avg_time = sum(times) / len(times) if times else 0
            avg_similarity = sum(similarities) / len(similarities) * 100 if similarities else 0
            avg_compression = sum(compressions) / len(compressions) * 100 if compressions else 0
            
            print(f"{model_name:<12} {success_rate:<15.1f}% {avg_time:<15.3f} {avg_similarity:<15.1f}% {avg_compression:<15.1f}%")
            
            model_stats[model_name] = {
                "success_rate": success_rate,
                "avg_time": avg_time,
                "avg_similarity": avg_similarity,
                "avg_compression": avg_compression
            }
    
    # Table 2: Detailed Results
    print("\n📊 TABLE 2: DETAILED RESULTS BY TEXT LENGTH")
    print("-" * 100)
    print(f"{'Length':<10} {'Model':<12} {'Success':<8} {'Time (s)':<10} {'Similarity':<12} {'Compression':<12} {'Words':<8}")
    print("-" * 100)
    
    for length_name, model_results in results.items():
        for model_name, result in model_results.items():
            if result and result.get("success"):
                success = "✓" if result["success"] else "✗"
                time_val = f"{result['processing_time']:.3f}"
                similarity = f"{result['semantic_similarity']*100:.1f}%"
                compression = f"{result['compression_ratio']*100:.1f}%"
                words = f"{result['word_count_summary']}"
                
                print(f"{length_name:<10} {model_name:<12} {success:<8} {time_val:<10} {similarity:<12} {compression:<12} {words:<8}")
    
    # Table 3: Best Model Recommendations
    print("\n📊 TABLE 3: BEST MODEL RECOMMENDATIONS")
    print("-" * 60)
    
    if model_stats:
        # Best for speed
        fastest_model = min(model_stats.items(), key=lambda x: x[1]["avg_time"])
        print(f"🏃 Fastest Model: {fastest_model[0]} ({fastest_model[1]['avg_time']:.3f}s)")
        
        # Best for accuracy
        most_accurate = max(model_stats.items(), key=lambda x: x[1]["avg_similarity"])
        print(f"🎯 Most Accurate: {most_accurate[0]} ({most_accurate[1]['avg_similarity']:.1f}% similarity)")
        
        # Best for compression
        best_compression = min(model_stats.items(), key=lambda x: x[1]["avg_compression"])
        print(f"📝 Best Compression: {best_compression[0]} ({best_compression[1]['avg_compression']:.1f}% compression)")
        
        # Overall best
        balanced_scores = {}
        for model, stats in model_stats.items():
            # Simple balanced score (lower is better for time and compression)
            balanced_score = (stats["avg_time"] * 0.2 + (100 - stats["avg_similarity"]) * 0.6 + stats["avg_compression"] * 0.2)
            balanced_scores[model] = balanced_score
        
        best_overall = min(balanced_scores.items(), key=lambda x: x[1])
        print(f"🏆 Best Overall: {best_overall[0]} (score: {best_overall[1]:.3f})")
    
    return model_stats

def save_results(results, model_stats):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_test_results_{timestamp}.json"
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_stats": model_stats,
        "detailed_results": results,
        "test_texts": {k: len(v) for k, v in test_texts.items()}
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {filename}")
    return filename

if __name__ == "__main__":
    # Import required modules
    from sentence_transformers import util
    
    # Run tests
    results = test_model_performance()
    
    # Generate tables
    model_stats = generate_tables(results)
    
    # Save results
    filename = save_results(results, model_stats)
    
    print(f"\n✅ Testing completed!")
    print("📋 Use these tables for your PowerPoint presentation!") 