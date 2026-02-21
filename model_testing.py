#!/usr/bin/env python3
"""
Model Testing and Evaluation Script for Web Summarizer
Tests accuracy, performance, and generates comparison tables
"""

import time
import json
import statistics
from datetime import datetime
from transformers import pipeline, AutoModelForSeq2SeqLM, BartTokenizer, T5Tokenizer, PegasusTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch

class ModelTester:
    def __init__(self):
        self.test_texts = {
            "short": [
                "Artificial intelligence is transforming the way we live and work. Machine learning algorithms can now process vast amounts of data to make predictions and automate complex tasks. This technology is being applied across industries from healthcare to finance.",
                "Climate change is one of the most pressing issues of our time. Rising global temperatures are causing extreme weather events, melting polar ice caps, and threatening biodiversity. Immediate action is needed to reduce greenhouse gas emissions.",
                "The internet has revolutionized communication and information sharing. Social media platforms connect billions of people worldwide, while e-commerce has transformed how we shop and do business."
            ],
            "medium": [
                """The COVID-19 pandemic has fundamentally changed the global landscape in unprecedented ways. When the virus first emerged in late 2019, few could have predicted the scale of its impact. Governments worldwide implemented lockdowns, travel restrictions, and social distancing measures to curb the spread. The healthcare system faced immense pressure as hospitals overflowed with patients. Economic activity ground to a halt, leading to widespread job losses and business closures. Remote work became the new normal for many industries, accelerating the adoption of digital technologies. Education systems had to rapidly adapt to online learning platforms. The pandemic also highlighted existing inequalities in healthcare access and economic stability. While vaccines have provided hope for recovery, the long-term effects on society, economy, and mental health continue to unfold. The experience has taught valuable lessons about global cooperation, scientific communication, and the importance of resilient healthcare systems.""",
                
                """Renewable energy sources are becoming increasingly important in the global transition toward sustainable energy systems. Solar power has seen remarkable growth, with photovoltaic technology becoming more efficient and cost-effective. Wind energy, both onshore and offshore, provides clean electricity to millions of homes worldwide. Hydropower continues to be a reliable source of renewable energy, particularly in regions with suitable geography. The development of energy storage technologies, such as advanced batteries, is crucial for addressing the intermittent nature of renewable sources. Governments are implementing policies and incentives to accelerate the adoption of clean energy technologies. The private sector is also investing heavily in renewable energy projects, recognizing both environmental and economic benefits. However, challenges remain, including the need for infrastructure upgrades, grid modernization, and addressing concerns about land use and wildlife impacts. The transition to renewable energy is not just about environmental protection but also about energy security and economic competitiveness.""",
                
                """Digital transformation is reshaping industries across the globe, driven by advances in technology and changing consumer expectations. Cloud computing has enabled businesses to scale operations efficiently while reducing infrastructure costs. Big data analytics provides insights that drive strategic decision-making and improve customer experiences. Artificial intelligence and machine learning automate routine tasks and create new capabilities. The Internet of Things connects devices and systems, enabling real-time monitoring and control. Cybersecurity has become paramount as organizations digitize their operations and face increasing threats. Mobile technology has changed how consumers interact with businesses, requiring responsive and user-friendly digital interfaces. E-commerce platforms have transformed retail, while digital payment systems have revolutionized financial transactions. The pandemic accelerated digital adoption, making technology essential for business continuity. Organizations must balance innovation with security, privacy, and ethical considerations in their digital transformation journeys."""
            ],
            "long": [
                """The rapid advancement of artificial intelligence and machine learning technologies is fundamentally transforming various sectors of society, from healthcare and education to transportation and entertainment. These technologies, which enable computers to learn from data and make decisions with minimal human intervention, are becoming increasingly sophisticated and accessible. In healthcare, AI systems can analyze medical images to detect diseases like cancer at earlier stages than traditional methods, potentially saving countless lives. Machine learning algorithms can predict patient outcomes, optimize treatment plans, and even assist in drug discovery by analyzing vast datasets of molecular structures and clinical trials. The education sector is also experiencing significant changes, with AI-powered tutoring systems providing personalized learning experiences tailored to individual student needs and learning styles. These systems can adapt their teaching methods based on student performance, offering additional support where needed and accelerating progress in areas where students excel. Transportation is being revolutionized by autonomous vehicles, which use complex AI systems to navigate roads, interpret traffic signals, and make split-second decisions to ensure passenger safety. While fully autonomous vehicles are still in development, many modern cars already incorporate AI features like adaptive cruise control and lane-keeping assistance. The entertainment industry has embraced AI for content creation, recommendation systems, and even scriptwriting assistance. Streaming platforms use sophisticated algorithms to suggest content based on viewing history and preferences, while AI tools help creators generate music, art, and written content. However, the widespread adoption of AI also raises important ethical and societal questions. Concerns about job displacement as automation replaces human workers in various industries are growing, requiring careful consideration of retraining programs and social safety nets. Privacy issues arise as AI systems collect and analyze vast amounts of personal data, necessitating robust regulations and ethical guidelines. Algorithmic bias, where AI systems perpetuate or amplify existing social inequalities, is another critical concern that must be addressed through diverse training data and inclusive design practices. The development of AI also raises questions about accountability and transparency, particularly in high-stakes applications like criminal justice and healthcare. As these technologies continue to evolve, it is essential to balance innovation with responsible development, ensuring that AI benefits society as a whole while minimizing potential harms. This requires collaboration between technologists, policymakers, ethicists, and the public to establish appropriate guidelines and regulations.""",
                
                """Climate change represents one of the most significant challenges facing humanity in the 21st century, with far-reaching implications for ecosystems, economies, and human societies worldwide. The scientific consensus is clear: human activities, particularly the burning of fossil fuels and deforestation, are driving unprecedented changes in Earth's climate system. Global temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times, with the rate of warming accelerating in recent decades. This warming is causing widespread and sometimes irreversible changes to natural systems, including melting glaciers and polar ice caps, rising sea levels, and shifting weather patterns. The impacts of climate change are already being felt across the globe, with more frequent and intense extreme weather events such as hurricanes, droughts, floods, and heatwaves. These events cause significant damage to infrastructure, agriculture, and human settlements, often disproportionately affecting vulnerable communities and developing nations. The economic costs of climate change are substantial, with estimates suggesting that unchecked warming could reduce global GDP by significant percentages by the end of the century. Agricultural systems are particularly vulnerable, as changing temperature and precipitation patterns affect crop yields and livestock productivity. Many regions are experiencing water scarcity, while others face increased flooding, creating complex challenges for water management and food security. Biodiversity is also under threat, with many species struggling to adapt to rapidly changing environmental conditions. Coral reefs, which support approximately 25% of marine life, are particularly vulnerable to ocean acidification and warming waters. The loss of biodiversity not only affects ecosystems but also has implications for human well-being, as many communities depend on natural resources for their livelihoods and cultural practices. Addressing climate change requires coordinated global action across multiple sectors. The transition to renewable energy sources, such as solar, wind, and hydropower, is essential for reducing greenhouse gas emissions from the energy sector. Energy efficiency improvements in buildings, transportation, and industry can significantly reduce energy demand and associated emissions. Sustainable land use practices, including reforestation and improved agricultural methods, can help sequester carbon and enhance ecosystem resilience. Technological innovations, such as carbon capture and storage, electric vehicles, and smart grid systems, offer additional tools for reducing emissions and adapting to climate impacts. However, the scale and urgency of the climate challenge require unprecedented levels of international cooperation and political will. The Paris Agreement, adopted in 2015, represents a significant step forward in global climate governance, with nearly all countries committing to reduce emissions and adapt to climate impacts. However, current commitments are insufficient to meet the agreement's goal of limiting warming to well below 2 degrees Celsius, let alone the more ambitious target of 1.5 degrees. Strengthening these commitments and ensuring their implementation requires continued diplomatic efforts and domestic policy action. The transition to a low-carbon economy also presents opportunities for sustainable development, job creation, and improved public health. Renewable energy industries are growing rapidly, creating employment opportunities in manufacturing, installation, and maintenance. Energy efficiency improvements can reduce energy costs for households and businesses while improving indoor air quality and comfort. Sustainable transportation options, such as public transit, cycling, and walking, can reduce air pollution and improve public health outcomes. The challenge of climate change is immense, but so too is human capacity for innovation, cooperation, and adaptation. Successfully addressing this challenge requires recognizing both the urgency of action and the opportunities for creating a more sustainable and equitable future. This will require unprecedented levels of collaboration between governments, businesses, civil society, and individuals, as well as a fundamental rethinking of how we produce and consume energy, food, and other resources. The decisions made in the coming years will have profound implications for future generations and the health of our planet."""
            ]
        }
        
        # Initialize models
        self.models = {}
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.load_models()
        
    def load_models(self):
        """Load all available models"""
        print("Loading models for testing...")
        
        # English models
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
        
        for key, model_name in english_models.items():
            try:
                print(f"Loading {key} model...")
                tokenizer_class = tokenizer_classes[key]
                tokenizer = tokenizer_class.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.models[key] = pipeline("summarization", model=model, tokenizer=tokenizer)
                print(f"✓ {key} loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {key}: {str(e)}")
                self.models[key] = None
        
        # Arabic model
        try:
            print("Loading Arabic model...")
            arabic_model_name = "csebuetnlp/mT5_multilingual_XLSum"
            arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model_name)
            arabic_model = AutoModelForSeq2SeqLM.from_pretrained(arabic_model_name)
            self.models["arabic"] = pipeline("summarization", model=arabic_model, tokenizer=arabic_tokenizer)
            print("✓ Arabic model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Arabic model: {str(e)}")
            self.models["arabic"] = None
    
    def test_model(self, model_name, text, max_length=150, min_length=50):
        """Test a single model on given text"""
        if self.models[model_name] is None:
            return None
            
        start_time = time.time()
        try:
            result = self.models[model_name](text, max_length=max_length, min_length=min_length)
            summary = result[0]["summary_text"]
            processing_time = time.time() - start_time
            
            # Calculate metrics
            compression_ratio = len(summary) / len(text)
            word_count_original = len(text.split())
            word_count_summary = len(summary.split())
            
            # Calculate semantic similarity
            embeddings = self.embedder.encode([text, summary], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            
            return {
                "summary": summary,
                "processing_time": processing_time,
                "compression_ratio": compression_ratio,
                "word_count_original": word_count_original,
                "word_count_summary": word_count_summary,
                "semantic_similarity": similarity,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def run_comprehensive_test(self):
        """Run comprehensive testing on all models"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL TESTING")
        print("="*60)
        
        results = {}
        
        for length_category, texts in self.test_texts.items():
            print(f"\nTesting {length_category.upper()} texts...")
            results[length_category] = {}
            
            for i, text in enumerate(texts):
                print(f"  Text {i+1}: {len(text)} characters")
                results[length_category][f"text_{i+1}"] = {}
                
                for model_name in self.models.keys():
                    if self.models[model_name] is not None:
                        result = self.test_model(model_name, text)
                        results[length_category][f"text_{i+1}"][model_name] = result
        
        return results
    
    def generate_comparison_tables(self, results):
        """Generate comparison tables from test results"""
        print("\n" + "="*60)
        print("MODEL COMPARISON TABLES")
        print("="*60)
        
        # Table 1: Overall Performance Summary
        print("\n📊 TABLE 1: OVERALL PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"{'Model':<12} {'Success Rate':<15} {'Avg Time (s)':<15} {'Avg Similarity':<15} {'Avg Compression':<15}")
        print("-" * 80)
        
        model_stats = {}
        for model_name in self.models.keys():
            if self.models[model_name] is None:
                continue
                
            success_count = 0
            total_count = 0
            times = []
            similarities = []
            compressions = []
            
            for category in results.values():
                for text_results in category.values():
                    if model_name in text_results:
                        result = text_results[model_name]
                        total_count += 1
                        if result and result.get("success"):
                            success_count += 1
                            times.append(result["processing_time"])
                            similarities.append(result["semantic_similarity"])
                            compressions.append(result["compression_ratio"])
            
            if total_count > 0:
                success_rate = (success_count / total_count) * 100
                avg_time = statistics.mean(times) if times else 0
                avg_similarity = statistics.mean(similarities) * 100 if similarities else 0
                avg_compression = statistics.mean(compressions) * 100 if compressions else 0
                
                print(f"{model_name:<12} {success_rate:<15.1f}% {avg_time:<15.3f} {avg_similarity:<15.1f}% {avg_compression:<15.1f}%")
                
                model_stats[model_name] = {
                    "success_rate": success_rate,
                    "avg_time": avg_time,
                    "avg_similarity": avg_similarity,
                    "avg_compression": avg_compression
                }
        
        # Table 2: Detailed Results by Text Length
        print("\n📊 TABLE 2: DETAILED RESULTS BY TEXT LENGTH")
        print("-" * 100)
        print(f"{'Length':<10} {'Model':<12} {'Success':<8} {'Time (s)':<10} {'Similarity':<12} {'Compression':<12} {'Words':<8}")
        print("-" * 100)
        
        for length_category, texts_data in results.items():
            for text_name, model_results in texts_data.items():
                for model_name, result in model_results.items():
                    if result and result.get("success"):
                        success = "✓" if result["success"] else "✗"
                        time_val = f"{result['processing_time']:.3f}"
                        similarity = f"{result['semantic_similarity']*100:.1f}%"
                        compression = f"{result['compression_ratio']*100:.1f}%"
                        words = f"{result['word_count_summary']}"
                        
                        print(f"{length_category:<10} {model_name:<12} {success:<8} {time_val:<10} {similarity:<12} {compression:<12} {words:<8}")
        
        # Table 3: Best Model Recommendations
        print("\n📊 TABLE 3: BEST MODEL RECOMMENDATIONS")
        print("-" * 60)
        
        if model_stats:
            # Best for speed
            fastest_model = min(model_stats.items(), key=lambda x: x[1]["avg_time"])
            print(f"🏃 Fastest Model: {fastest_model[0]} ({fastest_model[1]['avg_time']:.3f}s)")
            
            # Best for accuracy (semantic similarity)
            most_accurate = max(model_stats.items(), key=lambda x: x[1]["avg_similarity"])
            print(f"🎯 Most Accurate: {most_accurate[0]} ({most_accurate[1]['avg_similarity']:.1f}% similarity)")
            
            # Best for compression
            best_compression = min(model_stats.items(), key=lambda x: x[1]["avg_compression"])
            print(f"📝 Best Compression: {best_compression[0]} ({best_compression[1]['avg_compression']:.1f}% compression)")
            
            # Overall best (balanced score)
            balanced_scores = {}
            for model, stats in model_stats.items():
                # Normalize scores (0-1) and create balanced score
                time_score = 1 - (stats["avg_time"] / max(s["avg_time"] for s in model_stats.values()))
                similarity_score = stats["avg_similarity"] / 100
                compression_score = 1 - (stats["avg_compression"] / 100)  # Lower compression is better
                
                balanced_score = (time_score * 0.2 + similarity_score * 0.6 + compression_score * 0.2)
                balanced_scores[model] = balanced_score
            
            best_overall = max(balanced_scores.items(), key=lambda x: x[1])
            print(f"🏆 Best Overall: {best_overall[0]} (score: {best_overall[1]:.3f})")
        
        return model_stats
    
    def save_results(self, results, model_stats):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_test_results_{timestamp}.json"
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "model_stats": model_stats,
            "detailed_results": results,
            "test_config": {
                "total_texts": sum(len(texts) for texts in self.test_texts.values()),
                "text_categories": list(self.test_texts.keys()),
                "models_tested": list(self.models.keys())
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename

def main():
    """Main testing function"""
    print("🤖 Web Summarizer Model Testing Suite")
    print("=" * 50)
    
    tester = ModelTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Generate comparison tables
    model_stats = tester.generate_comparison_tables(results)
    
    # Save results
    filename = tester.save_results(results, model_stats)
    
    print(f"\n✅ Testing completed! Check '{filename}' for detailed results.")
    print("\n📋 Summary:")
    print("- Use the tables above to compare model performance")
    print("- Consider your specific needs (speed vs accuracy vs compression)")
    print("- Test with your own texts for domain-specific performance")

if __name__ == "__main__":
    main() 