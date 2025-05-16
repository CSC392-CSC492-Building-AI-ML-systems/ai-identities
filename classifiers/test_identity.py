import json
import numpy as np
from scipy.spatial.distance import cosine
from model_identity import ModelVisualizerManager

# Gotta work on this quite a lot, just a template from bullet points
# model_identity.py works great the way we have it right now, this is just
# to speed up the process of testing the identification system
class ModelIdentityTester:
    def __init__(self, manager):
        self.manager = manager
    
    def _cosine_similarity(self, vec1, vec2):
        return 1 - cosine(vec1, vec2)
    
    def _get_fingerprint_vector(self, word_freq, top_n=30):
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {word: freq for word, freq in top_words}

    def test_id_consistency(self, base_file, variant_file, model_name, temp1=1, temp2=0.8):
        """Test if similar temperature variants get same base ID"""
        # Load sample data
        with open(base_file) as f:
            base_data = json.load(f)
        with open(variant_file) as f:
            variant_data = json.load(f)

        # Store initial run
        base_record = self.manager.save_model_run(base_data, model_name, temp1)
        print(f"Initial ID: {base_record.id}")
        print(f"Base Temperature Range: {base_record.temperature}")

        # Store variant run
        variant_record = self.manager.save_model_run(variant_data, model_name, temp2)
        
        # Calculate drift score
        vec1 = self._get_fingerprint_vector(base_data)
        vec2 = self._get_fingerprint_vector(variant_data)
        drift_score = self._cosine_similarity(
            list(vec1.values()), 
            list(vec2.values())
        )
        
        print(f"\nVariant ID: {variant_record.id}")
        print(f"Drift Score: {drift_score:.4f}")
        print(f"Variant Temperature: {variant_record.temperature}")

        # Get database status
        report = self.manager.get_drift_report(model_name)
        print(f"\nFinal Temperature Range: {report['temperature_range']}")
        print(f"Total Runs: {report['runs']}")

    def test_model_identification(self, json_file, expected_model_name):
        """Test model name extraction and ID generation"""
        with open(json_file) as f:
            data = json.load(f)
        
        record = self.manager.save_model_run(data, expected_model_name)
        print(f"\nTest Model: {json_file}")
        print(f"Expected Name: {expected_model_name}")
        print(f"Stored Name: {record.model_name}")
        print(f"Generated ID: {record.id}")
        
        # Verify fingerprint in analysis data
        top_words = list(record.analysis_data['normalized_frequencies'].keys())[:5]
        print(f"Top 5 Words: {top_words}")

if __name__ == "__main__":
    # Initialize system with test database
    manager = ModelVisualizerManager(db_uri='postgresql://user:pass@localhost/test_db')
    tester = ModelIdentityTester(manager)

    print("=== Model Identification Test ===")
    tester.test_model_identification(
        json_file="gpt-4.5-preview_results.json",
        expected_model_name="GPT-4.5-Preview"
    )

    print("\n=== Temperature Drift Test ===")
    tester.test_id_consistency(
        base_file="classifiers/testing_IDS/t1_DeepSeek-R1-Turbo_results.json",
        variant_file="classifiers/testing_IDS/t2_DeepSeek-R1-Turbo_results.json",
        model_name="DeepSeek-R1-Turbo",
        temp1=1,
        temp2=0.5
    )