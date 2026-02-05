# -*- coding: utf-8 -*-
"""
Model Definitions
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class ModelFactory:
    """Factory class for creating ML models"""
    
    @staticmethod
    def get_models():
        """Return dictionary of all models"""
        models = {
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=1.0,
                max_depth=1,
                random_state=42
            ),
            'Naive Bayes': GaussianNB(),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(180, 280),
                max_iter=1000,
                learning_rate_init=0.003,
                random_state=42,
                early_stopping=True,
                n_iter_no_change=10
            )
        }
        return models
    
    @staticmethod
    def get_model_names():
        """Return list of model names"""
        return list(ModelFactory.get_models().keys())
