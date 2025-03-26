import json
import pickle
import uuid
from datetime import datetime
import numpy as np
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, JSON, Float, Text
from sqlalchemy.orm import declarative_base  # Updated import to avoid deprecation warning
from sqlalchemy.orm import sessionmaker
import os
import math
from scipy.spatial.distance import cosine

from visualizer import compare_models  # Direct visualizer integration
from heatMap_prelim_classifier import predict_model

Base = declarative_base()

class ModelIdentity(Base):
    __tablename__ = 'model_identities'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, index=True)
    base_fingerprint = Column(JSON)
    current_fingerprint = Column(JSON)
    statistical_profile = Column(JSON)
    classifier_data = Column(JSON)
    visualizations = Column(JSON)
    created_at = Column(sa.DateTime, default=datetime.now)
    drift_history = Column(JSON)

class UnifiedModelManager:
    def __init__(self, db_uri="postgresql://postgres:PostgresHuh@localhost/model_DB", vis_dir="visualizations"):
        self.engine = create_engine(db_uri)
        self.vis_dir = vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.classifier = self._load_classifier()
        self.drift_threshold = 0.92

    def _load_classifier(self):
        with open('classifiers/mlp_classifier.pkl', 'rb') as f:
            return pickle.load(f)

    def _create_fingerprint(self, word_freq, top_n=30):
        total = sum(word_freq.values())
        return {word: count/total for word, count in 
                sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]}

    def _calculate_stats(self, word_freq):
        counts = list(word_freq.values())
        sorted_counts = sorted(counts, reverse=True)
        total = sum(counts)
        
        # Entropy calculation
        probs = [c/total for c in counts if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        
        # Gini coefficient
        # source: stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
        gini = 1 - sum((c/total)**2 for c in counts)
        
        return {
            'entropy': entropy,
            'gini': gini,
            'top10_ratio': sum(sorted_counts[:10])/total,
            'temp_sensitivity': (sorted_counts[0] - sorted_counts[1])/total if len(sorted_counts) >1 else 0
        }

    
    def _generate_visualizations(self, word_freq, model_id):
        temp_json = f"temp_{model_id}.json"
        with open(temp_json, 'w') as f:
            json.dump(word_freq, f)
        
        compare_models([temp_json], top_n=30)
        
        prefix = f"{model_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        vis_paths = {
            'heatmap': os.path.join(self.vis_dir, f"{prefix}_heatmap.png"),
            'top_words': os.path.join(self.vis_dir, f"{prefix}_top_words.png")
        }
        
        os.rename('top_words_by_model.png', vis_paths['top_words'])
        os.rename('word_frequency_heatmap.png', vis_paths['heatmap'])
        
        if os.path.exists('model_pca.png'):
            vis_paths['pca'] = os.path.join(self.vis_dir, f"{prefix}_pca.png")
            os.rename('model_pca.png', vis_paths['pca'])
        
        return vis_paths

    def process_model_run(self, word_freq):
        """Full processing pipeline for new data"""
        session = self.Session()
        
        # Define result values to return after session closes
        record_id = None
        record_drift_history = []
        record_stats = {}
        add_new_record = True  # Flag to control whether to add new record
        
        try:
            # Get model prediction from our good ol classifier
            classifier_pred = predict_model(self.classifier, word_freq)
            
            # IMP: Extra file for now, a copy of the word freq created in generate_visualization, gets removed later
            temp_json = f"temp_{classifier_pred}.json"
            fingerprint = self._create_fingerprint(word_freq)
            stats = self._calculate_stats(word_freq)
            
            # Generate visualizations
            # TODO: Can be made so that we dont need to generate visualizations, have done that
            # TODO: extensively in the past, but for now, I can't remove it since temp_json is created
            # TODO: in generate_visualizations function, and needed later
            vis_paths = self._generate_visualizations(word_freq, classifier_pred)
            
            # IMP: Check if the predicted model already exists in the database
            existing = session.query(ModelIdentity).filter(
                ModelIdentity.model_name == classifier_pred
            ).order_by(ModelIdentity.created_at.desc()).first()

            new_record = ModelIdentity(
                model_name=classifier_pred,
                classifier_data=classifier_pred,
                statistical_profile=stats,
                current_fingerprint=fingerprint,
                base_fingerprint=fingerprint,  # Initalizing step, may or may not be updated depending on drift
                visualizations=vis_paths,
                drift_history=[]
            )

            if existing:
                if existing.base_fingerprint:  # Sanity check
                    # Drift
                    drift_score = 1 - cosine(
                        list(existing.base_fingerprint.values()),
                        list(fingerprint.values())
                    )
                    
                    # If drift score is above threshold, we keep the existing model
                    # and just update its drift history
                    if drift_score >= self.drift_threshold:
                        print(f"Low drift detected ({drift_score:.4f}): Updating existing model rather than creating new one")
                        # Updating the existing model record's drift history here
                        existing.drift_history = existing.drift_history + [drift_score] if existing.drift_history else [drift_score]
                        # Update the record but don't add a new one
                        session.add(existing)
                        add_new_record = False
                        
                        # Set return values to use existing model's data
                        record_id = existing.id
                        record_drift_history = existing.drift_history.copy() if existing.drift_history else []
                        record_stats = dict(existing.statistical_profile) if existing.statistical_profile else {}
                    else:
                        # High drift - create new model with new ID but link drift history
                        print(f"High drift detected ({drift_score:.4f}): Creating new model variant")
                        new_record.drift_history = existing.drift_history + [drift_score] if existing.drift_history else [drift_score]
                        new_record.base_fingerprint = fingerprint  # New base fingerprint for this variant
                        new_record.id = str(uuid.uuid4())  # Generate new ID
                else:
                    # Handle case where existing model doesn't have base_fingerprint
                    new_record.drift_history = existing.drift_history if existing.drift_history else []
                    new_record.base_fingerprint = fingerprint  # Use current fingerprint as base

            
            # TODO: Maybe make it so that since this was predicted as the same model, \
            # TODO: but somehow has a different drift score (look into the why), we say that its a "finetuned"
            # TODO: version of <existing model name>.
            # Add the new reocrd to the session
            if add_new_record:
                session.add(new_record)
                record_id = new_record.id
                record_drift_history = new_record.drift_history.copy() if new_record.drift_history else []
                record_stats = dict(new_record.statistical_profile) if new_record.statistical_profile else {}
            
            session.commit()
            
            return {
                'id': record_id,
                'drift_history': record_drift_history,
                'stats': record_stats
            }

        finally:
            session.close()
            # Remove the temporary file now
            for f in [temp_json, 'heatmap_data_2.json']:
                if os.path.exists(f):
                    os.remove(f)

class EnhancedClassifier:
    def __init__(self, manager):
        self.manager = manager
        self.classifier = manager.classifier

    def predict_with_context(self, word_freq):
        base_pred = predict_model(self.classifier, word_freq)
        record_data = self.manager.process_model_run(word_freq)
        
        return {
            'prediction': base_pred,
            'model_id': record_data['id'],
            'drift_score': record_data['drift_history'][-1] if record_data['drift_history'] else 1.0,
            'stats': record_data['stats']
        }

# Usage example
if __name__ == "__main__":
    manager = UnifiedModelManager()
    enhanced_clf = EnhancedClassifier(manager)
    
    with open('classifiers/testing_IDS/t1_DeepSeek-R1-Turbo_results.json') as f:
        data = json.load(f)
    
    result = enhanced_clf.predict_with_context(data)
    print(f"Predicted: {result['prediction']}")
    print(f"Model ID: {result['model_id']}")
    print(f"Drift Score: {result['drift_score']:.4f}")