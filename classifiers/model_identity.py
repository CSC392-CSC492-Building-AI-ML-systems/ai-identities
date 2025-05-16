import json
import pickle
import uuid
from datetime import datetime
import numpy as np
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, JSON, Float, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import math
from scipy.spatial.distance import cosine, jaccard, cityblock
import scipy.stats as stats

from visualizer import compare_models
from heatMap_prelim_classifier import predict_model

Base = declarative_base()

class EnhancedModelIdentity(Base):
    __tablename__ = 'enhanced_model_identities'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, index=True)
    base_fingerprint = Column(JSON)
    current_fingerprint = Column(JSON)
    statistical_profile = Column(JSON)
    drift_metrics = Column(JSON)
    distinctive_words = Column(JSON)
    created_at = Column(sa.DateTime, default=datetime.now)
    drift_history = Column(JSON)

class EnhancedModelManager:
    def __init__(self, db_uri="postgresql://postgres:PostgresHuh@localhost/model_DB", 
                 vis_dir="visualizations", drift_sensitivity=0.75):
        self.engine = create_engine(db_uri)
        self.vis_dir = vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.classifier = self._load_classifier()
        self.drift_sensitivity = drift_sensitivity

    def _load_classifier(self):
        with open('classifiers/mlp_classifier.pkl', 'rb') as f:
            return pickle.load(f)

    def _create_advanced_fingerprint(self, word_freq, top_n=50):
        """
        Create a multi-dimensional fingerprint with additional metrics
        """
        total = sum(word_freq.values())
        
        # Basic frequency normalization
        normalized_freq = {word: count/total for word, count in 
                sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]}
        
        # Rarity scoring (inverse frequency)
        rarity_scores = {}
        total_words = sum(word_freq.values())
        for word, count in word_freq.items():
            occurrence_prob = count / total_words
            rarity_scores[word] = -math.log(occurrence_prob + 1e-10)
        
        # Log frequency to capture power law distribution
        log_freq = {word: math.log(count + 1) for word, count in word_freq.items()}
        
        # Distinctive word identification
        distinctive_words = {}
        threshold = np.percentile(list(rarity_scores.values()), 90)
        for word, rarity in rarity_scores.items():
            if rarity > threshold:
                distinctive_words[word] = {
                    'count': word_freq.get(word, 0),
                    'rarity_score': rarity
                }
        
        # Compute Zipf's law fit
        sorted_counts = sorted(word_freq.values(), reverse=True)
        rank_freq = [(rank+1, freq) for rank, freq in enumerate(sorted_counts)]
        log_rank = np.log([r for r, _ in rank_freq])
        log_freq = np.log([f for _, f in rank_freq])
        
        try:
            slope, intercept, r_value, _, _ = stats.linregress(log_rank, log_freq)
            zipf_fit = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2
            }
        except Exception:
            zipf_fit = None
        
        return {
            'normalized_freq': normalized_freq,
            'rarity_scores': rarity_scores,
            'log_freq': log_freq,
            'distinctive_words': distinctive_words,
            'zipf_fit': zipf_fit,
            'hapax_legomena_ratio': self._compute_hapax_legomena(word_freq)
        }

    def _compute_hapax_legomena(self, word_freq):
        """
        Compute the ratio of words that appear only once (hapax legomena)
        """
        total_words = sum(word_freq.values())
        hapax_words = sum(1 for count in word_freq.values() if count == 1)
        return hapax_words / total_words if total_words > 0 else 0

    def _compute_drift_metrics(self, base_fingerprint, current_fingerprint):
        """
        Compute multiple drift metrics with weighted combination
        """
        # Cosine similarity on normalized frequencies
        cosine_dist = 1 - cosine(
            list(base_fingerprint['normalized_freq'].values())[:15], 
            list(current_fingerprint['normalized_freq'].values())[:15]
        )
        
        # Manhattan distance on rarity scores
        manhattan_dist = 1 - cityblock(
            list(base_fingerprint['rarity_scores'].values())[:15],
            list(current_fingerprint['rarity_scores'].values())[:15]
        )
        
        # Jaccard similarity on distinctive words
        base_distinct_words = set(base_fingerprint['distinctive_words'].keys())
        current_distinct_words = set(current_fingerprint['distinctive_words'].keys())
        jaccard_dist = jaccard(base_distinct_words, current_distinct_words)
        
        # Zipf's law fit comparison
        zipf_drift = abs(
            base_fingerprint['zipf_fit']['slope'] - 
            current_fingerprint['zipf_fit']['slope']
        ) if base_fingerprint['zipf_fit'] and current_fingerprint['zipf_fit'] else 0
        
        # Weighted drift score
        drift_score = (
            0.4 * cosine_dist + 
            0.3 * manhattan_dist + 
            0.2 * (1 - jaccard_dist) + 
            0.1 * (1 - zipf_drift)
        )
        
        return {
            'cosine_similarity': cosine_dist,
            'manhattan_distance': manhattan_dist,
            'jaccard_similarity': 1 - jaccard_dist,
            'zipf_drift': zipf_drift,
            'combined_drift_score': drift_score
        }

    def process_model_run(self, word_freq):
        """Enhanced processing pipeline for new data"""
        session = self.Session()
        
        try:
            # Classifier prediction
            classifier_pred = predict_model(self.classifier, word_freq)
            
            # Create advanced fingerprint
            current_fingerprint = self._create_advanced_fingerprint(word_freq)
            
            # Find existing model record
            existing = session.query(EnhancedModelIdentity).filter(
                EnhancedModelIdentity.model_name == classifier_pred
            ).order_by(EnhancedModelIdentity.created_at.desc()).first()

            # Compute drift metrics if base fingerprint exists
            drift_metrics = None
            if existing and existing.base_fingerprint:
                drift_metrics = self._compute_drift_metrics(
                    existing.base_fingerprint, 
                    current_fingerprint
                )
            
            # Determine if this is a new model variant
            is_new_variant = (
                not existing or 
                (drift_metrics and drift_metrics['combined_drift_score'] > self.drift_sensitivity)
            )
            
            # Create new record or update existing
            if is_new_variant:
                new_record = EnhancedModelIdentity(
                    model_name=classifier_pred,
                    base_fingerprint=current_fingerprint,
                    current_fingerprint=current_fingerprint,
                    statistical_profile=current_fingerprint,
                    drift_metrics=drift_metrics,
                    distinctive_words=current_fingerprint['distinctive_words'],
                    drift_history=[drift_metrics['combined_drift_score']] if drift_metrics else []
                )
                session.add(new_record)
                record_id = new_record.id
            else:
                existing.current_fingerprint = current_fingerprint
                existing.drift_metrics = drift_metrics
                if drift_metrics:
                    existing.drift_history.append(drift_metrics['combined_drift_score'])
                record_id = existing.id
            
            session.commit()
            
            return {
                'id': record_id,
                'model': classifier_pred,
                'drift_metrics': drift_metrics,
                'is_new_variant': is_new_variant
            }

        finally:
            session.close()

class EnhancedModelClassifier:
    def __init__(self, manager):
        self.manager = manager

    def analyze_model(self, word_freq):
        """Comprehensive model analysis with drift detection"""
        analysis = self.manager.process_model_run(word_freq)
        
        # Additional detailed analysis can be added here
        return {
            'model_prediction': analysis['model'],
            'model_id': analysis['id'],
            'is_new_model_variant': analysis['is_new_variant'],
            'drift_metrics': analysis['drift_metrics'] or {}
        }

# Example usage
if __name__ == "__main__":
    manager = EnhancedModelManager()
    enhanced_clf = EnhancedModelClassifier(manager)
    
    # Load some test data
    with open('classifiers/testing_IDS/Llama-3.2-90B-Vision-Instruct_results_1.0.json') as f:
        data = json.load(f)
    
    result = enhanced_clf.analyze_model(data)
    print(json.dumps(result, indent=2))