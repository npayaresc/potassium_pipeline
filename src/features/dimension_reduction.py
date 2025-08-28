"""
Modular Dimensionality Reduction Module

Provides multiple dimensionality reduction strategies that can be easily interchanged
through configuration without code changes.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

logger = logging.getLogger(__name__)


class DimensionReducer(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for all dimensionality reduction strategies."""
    
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the dimensionality reduction model."""
        pass
    
    @abstractmethod
    def transform(self, X):
        """Transform the data to reduced dimensions."""
        pass
    
    @abstractmethod
    def get_n_components(self):
        """Return the number of components after reduction."""
        pass
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: str):
        """Save the reducer to disk."""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a reducer from disk."""
        return joblib.load(filepath)


class PCAReducer(DimensionReducer):
    """Principal Component Analysis dimensionality reduction."""
    
    def __init__(self, n_components: Union[int, float] = 0.95, **kwargs):
        """
        Initialize PCA reducer.
        
        Args:
            n_components: Number of components or variance to retain (0-1)
            **kwargs: Additional arguments passed to sklearn PCA
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.pca = None
        self.n_features_ = None
        self.n_components_ = None
        
    def fit(self, X, y=None):
        """Fit PCA model."""
        self.pca = PCA(n_components=self.n_components, **self.kwargs)
        self.pca.fit(X)
        self.n_features_ = X.shape[1]
        self.n_components_ = self.pca.n_components_
        
        variance_retained = np.sum(self.pca.explained_variance_ratio_) * 100
        logger.info(f"PCA fitted: {self.n_features_} → {self.n_components_} components "
                   f"(retained {variance_retained:.1f}% variance)")
        return self
    
    def transform(self, X):
        """Transform data using fitted PCA."""
        if self.pca is None:
            raise ValueError("PCA must be fitted before transform")
        return self.pca.transform(X)
    
    def get_n_components(self):
        """Return number of components."""
        return self.n_components_ if self.n_components_ is not None else self.n_components


class PLSReducer(DimensionReducer):
    """Partial Least Squares dimensionality reduction."""
    
    def __init__(self, n_components: int = 10, scale: bool = True, **kwargs):
        """
        Initialize PLS reducer.
        
        Args:
            n_components: Number of components to extract
            scale: Whether to scale features before PLS
            **kwargs: Additional arguments passed to PLSRegression
        """
        self.n_components = n_components
        self.scale = scale
        self.kwargs = kwargs
        self.pls = None
        self.scaler = None
        self.n_features_ = None
        
    def fit(self, X, y=None):
        """Fit PLS model. Requires target values."""
        if y is None:
            raise ValueError("PLS requires target values (y) for fitting")
            
        # Scale features if requested
        if self.scale:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        # Fit PLS
        self.pls = PLSRegression(n_components=self.n_components, **self.kwargs)
        self.pls.fit(X_scaled, y)
        self.n_features_ = X.shape[1]
        
        logger.info(f"PLS fitted: {self.n_features_} → {self.n_components} components")
        return self
    
    def transform(self, X):
        """Transform data using fitted PLS."""
        if self.pls is None:
            raise ValueError("PLS must be fitted before transform")
            
        if self.scaler is not None:
            X = self.scaler.transform(X)
            
        # PLSRegression.transform returns X scores
        return self.pls.transform(X)
    
    def get_n_components(self):
        """Return number of components."""
        return self.n_components


class AutoencoderReducer(DimensionReducer):
    """Autoencoder-based nonlinear dimensionality reduction."""
    
    def __init__(self, n_components: int = 10, hidden_layers: Optional[list] = None,
                 epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                 device: str = 'auto', **kwargs):
        """
        Initialize Autoencoder reducer.
        
        Args:
            n_components: Size of the bottleneck layer
            hidden_layers: List of hidden layer sizes (encoder part)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to use ('cpu', 'cuda', or 'auto')
            **kwargs: Additional training parameters
        """
        self.n_components = n_components
        self.hidden_layers = hidden_layers or [64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.kwargs = kwargs
        
        self.model = None
        self.scaler = None
        self.n_features_ = None
        self._device = None
        
    def _get_device(self):
        """Determine the device to use."""
        if self._device is None:
            if self.device == 'auto':
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self._device = torch.device(self.device)
        return self._device
    
    def _build_model(self, input_dim: int):
        """Build the autoencoder architecture."""
        # Encoder layers
        encoder_layers = []
        prev_size = input_dim
        
        for hidden_size in self.hidden_layers:
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Bottleneck
        encoder_layers.append(nn.Linear(prev_size, self.n_components))
        
        # Decoder layers (mirror of encoder)
        decoder_layers = [nn.Linear(self.n_components, self.hidden_layers[-1])]
        
        for i in range(len(self.hidden_layers) - 1, 0, -1):
            decoder_layers.extend([
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_layers[i]),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_layers[i], self.hidden_layers[i-1])
            ])
        
        decoder_layers.extend([
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_layers[0]),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_layers[0], input_dim)
        ])
        
        # Create full autoencoder
        encoder = nn.Sequential(*encoder_layers)
        decoder = nn.Sequential(*decoder_layers)
        
        return nn.ModuleDict({'encoder': encoder, 'decoder': decoder})
    
    def fit(self, X, y=None):
        """Fit the autoencoder."""
        device = self._get_device()
        self.n_features_ = X.shape[1]
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        dataset = TensorDataset(X_tensor, X_tensor)  # Input and target are the same
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build and train model
        self.model = self._build_model(self.n_features_).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, _ in dataloader:
                # Forward pass
                encoded = self.model['encoder'](batch_X)
                decoded = self.model['decoder'](encoded)
                
                # Compute loss
                loss = criterion(decoded, batch_X)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Autoencoder epoch {epoch + 1}/{self.epochs}, "
                           f"reconstruction loss: {avg_loss:.4f}")
        
        self.model.eval()
        logger.info(f"Autoencoder fitted: {self.n_features_} → {self.n_components} components")
        return self
    
    def transform(self, X):
        """Transform data using the encoder part."""
        if self.model is None:
            raise ValueError("Autoencoder must be fitted before transform")
        
        device = self._get_device()
        self.model.eval()
        
        # Scale and convert to tensor
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Encode
        with torch.no_grad():
            encoded = self.model['encoder'](X_tensor)
        
        return encoded.cpu().numpy()
    
    def get_n_components(self):
        """Return number of components."""
        return self.n_components


class FeatureClusteringReducer(DimensionReducer):
    """Feature clustering-based dimensionality reduction."""
    
    def __init__(self, n_components: int = 10, method: str = 'kmeans', **kwargs):
        """
        Initialize Feature Clustering reducer.
        
        Args:
            n_components: Number of feature clusters (output dimensions)
            method: Clustering method ('kmeans' for now)
            **kwargs: Additional arguments for clustering
        """
        self.n_components = n_components
        self.method = method
        self.kwargs = kwargs
        
        self.feature_clusters_ = None
        self.cluster_centers_ = None
        self.n_features_ = None
        self.feature_labels_ = None
        
    def fit(self, X, y=None):
        """Fit feature clustering."""
        self.n_features_ = X.shape[1]
        
        # Transpose to cluster features (not samples)
        X_features = X.T
        
        if self.method == 'kmeans':
            # Use silhouette score to find optimal clusters if not specified
            if self.n_components == 'auto':
                best_score = -1
                best_n = 2
                # Use random_state from kwargs if provided, otherwise default to 42
                random_state = self.kwargs.get('random_state', 42)
                n_init = self.kwargs.get('n_init', 10)
                for n in range(2, min(21, self.n_features_ // 2)):
                    kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=n_init)
                    labels = kmeans.fit_predict(X_features)
                    score = silhouette_score(X_features, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
                self.n_components = best_n
                logger.info(f"Auto-selected {self.n_components} clusters based on silhouette score")
            
            # Perform clustering (random_state is extracted from kwargs or defaults to 42)
            clusterer = KMeans(n_clusters=self.n_components, **self.kwargs)
            self.feature_labels_ = clusterer.fit_predict(X_features)
            self.cluster_centers_ = clusterer.cluster_centers_
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        logger.info(f"Feature clustering fitted: {self.n_features_} → {self.n_components} clusters")
        return self
    
    def transform(self, X):
        """Transform data by selecting representative features from each cluster."""
        if self.feature_labels_ is None:
            raise ValueError("Feature clustering must be fitted before transform")
        
        # For each cluster, select the most representative feature
        # (closest to cluster center)
        X_reduced = np.zeros((X.shape[0], self.n_components))
        
        for cluster_id in range(self.n_components):
            # Find features in this cluster
            cluster_features = np.where(self.feature_labels_ == cluster_id)[0]
            
            if len(cluster_features) == 0:
                continue
            
            if len(cluster_features) == 1:
                # Only one feature in cluster
                X_reduced[:, cluster_id] = X[:, cluster_features[0]]
            else:
                # Take mean of all features in cluster
                X_reduced[:, cluster_id] = np.mean(X[:, cluster_features], axis=1)
        
        return X_reduced
    
    def get_n_components(self):
        """Return number of components."""
        return self.n_components


class DimensionReductionFactory:
    """Factory for creating dimensionality reduction instances."""
    
    _reducers = {
        'pca': PCAReducer,
        'pls': PLSReducer,
        'autoencoder': AutoencoderReducer,
        'feature_clustering': FeatureClusteringReducer,
    }
    
    @classmethod
    def create_reducer(cls, method: str, params: Dict[str, Any]) -> DimensionReducer:
        """
        Create a dimensionality reducer instance.
        
        Args:
            method: Name of the reduction method
            params: Parameters for the reducer
            
        Returns:
            DimensionReducer instance
        """
        if method not in cls._reducers:
            raise ValueError(f"Unknown reduction method: {method}. "
                           f"Available methods: {list(cls._reducers.keys())}")
        
        reducer_class = cls._reducers[method]
        return reducer_class(**params)
    
    @classmethod
    def register_reducer(cls, name: str, reducer_class):
        """Register a new reducer type."""
        if not issubclass(reducer_class, DimensionReducer):
            raise ValueError("Reducer class must inherit from DimensionReducer")
        cls._reducers[name] = reducer_class
    
    @classmethod
    def get_available_methods(cls) -> list:
        """Get list of available reduction methods."""
        return list(cls._reducers.keys())


# Auto-register advanced autoencoders when module is imported
try:
    from .advanced_autoencoders import register_advanced_autoencoders
    register_advanced_autoencoders()
except ImportError:
    # Advanced autoencoders not available
    pass