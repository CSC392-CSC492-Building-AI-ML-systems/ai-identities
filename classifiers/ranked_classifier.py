import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelDistinguishClassifier:
    def __init__(self, json_directory, max_samples_per_model=None):
        """
        Initialize the classifier focused on model distinction
        
        Args:
            json_directory (str): Path to directory containing model JSON files
            max_samples_per_model (int, optional): Limit samples per model to balance dataset
        """
        self.json_directory = json_directory
        self.max_samples_per_model = max_samples_per_model
        self.data = []
        self.labels = []
        self.model_names = []
    
    def _extract_model_name(self, filename):
        """
        Extract model name from filename, handling different naming patterns
        
        Args:
            filename (str): Name of the JSON file
        
        Returns:
            str: Extracted model name
        """
        # Remove file extension
        base_name = os.path.splitext(filename)[0]
        
        # Split by temperature or other separators
        parts = re.split(r'[_\-.]', base_name)
        
        # Return the first part (model name)
        return parts[0]
    
    def _load_and_preprocess_data(self):
        """
        Load JSON files and extract descriptions based on word frequencies
        """
        # Sort files to ensure consistent processing
        json_files = sorted([f for f in os.listdir(self.json_directory) if f.endswith('.json')])
        
        # Track samples per model to potentially limit
        model_sample_counts = {}
        
        for filename in json_files:
            # Extract model name using the new method
            model_name = self._extract_model_name(filename)
            
            # Load JSON file
            try:
                with open(os.path.join(self.json_directory, filename), 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {filename}. Skipping.")
                continue
            
            # Generate synthetic descriptions from word frequencies
            for position, word_freqs in file_data.items():
                # Create synthetic descriptions based on word frequencies
                for word, freq in word_freqs.items():
                    # Create multiple synthetic descriptions based on frequency
                    for _ in range(min(freq, 7)):  # Limit to 3 synthetic descriptions per word
                        self.data.append(f"This is a {word} description")
                        self.labels.append(model_name)
        
        print(f"Loaded {len(self.data)} descriptions across {len(set(self.labels))} models")
        print("Model breakdown:")
        for model in set(self.labels):
            print(f"  {model}: {self.labels.count(model)} descriptions")
        
        return self.data, self.labels
    
    def train_and_evaluate(self, 
                            n_splits=3, 
                            test_size=0.2, 
                            random_state=42, 
                            vectorizer_params=None):
        """
        Comprehensive model training and evaluation
        """
        # Load data
        X, y = self._load_and_preprocess_data()
        
        # Ensure we have data
        if len(X) == 0:
            raise ValueError("No descriptions found. Check your JSON files.")
        
        # Default vectorizer parameters
        default_vectorizer_params = {
            'max_features': 1000,
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'analyzer': 'word'
        }
        
        # Update with custom parameters if provided
        if vectorizer_params:
            default_vectorizer_params.update(vectorizer_params)
        
        # Vectorization
        vectorizer = TfidfVectorizer(**default_vectorizer_params)
        X_vectorized = vectorizer.fit_transform(X)
        
        # Classifier
        clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state, 
            class_weight='balanced',
            n_jobs=-2,
            verbose=1,
            max_depth=100
        )
        
        # Stratified K-Fold Cross-Validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Perform cross-validation
        cv_scores = cross_val_score(clf, X_vectorized, y, cv=cv, scoring='accuracy')
        
        # Fit on full dataset for final model and feature importance
        clf.fit(X_vectorized, y)
        
        # Get feature names
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            feature_names = vectorizer.get_feature_names()
        
        # Feature Importance
        feature_importance = self._get_top_features(clf, feature_names)
        
        # Confusion Matrix Visualization
        plt.figure(figsize=(10, 8))
        self._plot_confusion_matrix(clf, X_vectorized, y)
        
        # Prepare results
        results = {
            'cross_validation_scores': cv_scores.tolist(),
            'mean_cv_accuracy': np.mean(cv_scores),
            'std_cv_accuracy': np.std(cv_scores),
            'top_features': feature_importance,
            'unique_models': list(set(y))
        }
        
        return results
    
    def _get_top_features(self, clf, feature_names, top_n=20):
        """
        Extract top features for each model class
        """
        importances = clf.feature_importances_
        feature_importance = {}
        
        for i, model_name in enumerate(clf.classes_):
            # Sort features by importance
            sorted_idx = importances.argsort()[::-1]
            top_feature_indices = sorted_idx[:top_n]
            
            feature_importance[model_name] = [
                (feature_names[idx], importances[idx]) 
                for idx in top_feature_indices
            ]
        
        return feature_importance
    
    def _plot_confusion_matrix(self, clf, X, y):
        """
        Plot and save confusion matrix
        """
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred, labels=clf.classes_)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=clf.classes_, 
                    yticklabels=clf.classes_)
        plt.title('Confusion Matrix of Model Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('model_confusion_matrix.png')
        plt.close()

def main():
    # Specify the directory containing your JSON files
    json_directory = r'C:\Users\User\Github_academic\CSC392\ai-identities\classifiers\rankedJSON'
    
    # Initialize the classifier
    classifier = ModelDistinguishClassifier(
        json_directory, 
        max_samples_per_model=100  # Optional: limit samples per model
    )
    
    # Custom vectorizer parameters (optional)
    vectorizer_params = {
        'max_features': 1500,  # Adjust based on your data
        'ngram_range': (1, 3)  # Capture unigrams, bigrams, and trigrams
    }
    
    # Train and evaluate
    try:
        results = classifier.train_and_evaluate(
            n_splits=5,
            vectorizer_params=vectorizer_params
        )
        
        # Print results
        print("Model Distinction Results:")
        print(f"Mean Cross-Validation Accuracy: {results['mean_cv_accuracy']:.4f}")
        print(f"Standard Deviation of Accuracy: {results['std_cv_accuracy']:.4f}")
        
        # Save detailed results
        with open('model_distinction_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print top features for each model
        print("\nTop Features per Model:")
        for model, features in results['top_features'].items():
            print(f"\n{model}:")
            for feature, importance in features:
                print(f"  {feature}: {importance:.4f}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()