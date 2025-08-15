#!/usr/bin/env python3
"""
Test script for the enhanced dataset summarizer with feature analysis and histograms.
"""

import json
from dataset_sumarizer import summarize_dataset

def test_enhanced_summarizer():
    """Test the enhanced summarizer with a simple dataset."""
    
    print("Testing enhanced dataset summarizer...")
    
    try:
        # Test with a simple text dataset
        summary = summarize_dataset(
            "imdb", 
            config=None, 
            max_example_chars=200,
            analyze_features=True,
            max_analysis_samples=100  # Small sample for testing
        )
        
        print("✅ Successfully generated enhanced summary!")
        print(f"Dataset: {summary['repo_id']}")
        
        # Show feature analysis results
        for config_name, config_data in summary['by_config'].items():
            print(f"\n📊 Config: {config_name}")
            
            if config_data.get('feature_histogram'):
                histogram = config_data['feature_histogram']
                print(f"  Feature types: {histogram['feature_type_distribution']}")
                print(f"  Total features: {histogram['total_features']}")
                
                # Show feature details
                for feature in histogram['feature_details'][:5]:  # Show first 5
                    print(f"    - {feature['name']}: {feature['type']} "
                          f"(count: {feature['count']}, null: {feature['null_count']})")
            
            if config_data.get('feature_analysis'):
                print(f"\n  📈 Feature Analysis:")
                for feature_name, analysis in list(config_data['feature_analysis'].items())[:3]:  # Show first 3
                    print(f"    {feature_name}:")
                    print(f"      Type: {analysis['type']}")
                    print(f"      Count: {analysis['count']}")
                    
                    if analysis['type'] == 'text':
                        length_stats = analysis.get('length_stats', {})
                        print(f"      Text length - mean: {length_stats.get('mean', 'N/A')}, "
                              f"std: {length_stats.get('std', 'N/A')}")
                    
                    elif analysis['type'] == 'numerical':
                        stats = analysis.get('stats', {})
                        print(f"      Numerical - mean: {stats.get('mean', 'N/A')}, "
                              f"std: {stats.get('std', 'N/A')}")
                        
                        histogram = analysis.get('histogram', {})
                        if histogram.get('counts'):
                            print(f"      Histogram bins: {len(histogram['counts'])}")
        
        # Save full summary to file for inspection
        with open('enhanced_summary_output.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full summary saved to 'enhanced_summary_output.json'")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_summarizer()
