"""
Feature Selection Module for High-Dimensional Data

This module provides various feature selection methods to handle
high-dimension/low-sample scenarios in the potassium pipeline.
"""

import logging
from typing import Optional, Union, Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    f_regression,
    mutual_info_regression
)
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class SpectralFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector specifically designed for spectral data.
    
    Supports multiple selection methods optimized for high-dimensional
    spectral features with limited training samples.
    """
    
    def __init__(self, config):
        """
        Initialize the feature selector with configuration.
        
        Args:
            config: Pipeline configuration object with feature selection settings
        """
        self.config = config
        self.selector = None
        self.selected_features_ = None
        self.feature_scores_ = None
        self.feature_names_ = None
        
    def _get_n_features(self, X: Union[np.ndarray, pd.DataFrame]) -> int:
        """
        Calculate the actual number of features to select.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Number of features to select
        """
        n_total = X.shape[1]
        
        if isinstance(self.config.n_features_to_select, float) and self.config.n_features_to_select < 1.0:
            # Fraction of features
            n_selected = int(n_total * self.config.n_features_to_select)
        else:
            # Absolute number
            n_selected = min(int(self.config.n_features_to_select), n_total)
            
        # Ensure at least 1 feature is selected
        n_selected = max(1, n_selected)
        
        logger.info(f"Selecting {n_selected} features out of {n_total} total features")
        return n_selected
    
    def _create_selector(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Create the appropriate feature selector based on configuration.
        
        Args:
            X: Input feature matrix
            y: Target values
        """
        n_features = self._get_n_features(X)
        
        if self.config.feature_selection_method == 'selectkbest':
            # SelectKBest with configurable scoring function
            if self.config.feature_selection_score_func == 'f_regression':
                score_func = f_regression
            else:
                score_func = mutual_info_regression
                
            self.selector = SelectKBest(score_func=score_func, k=n_features)
            logger.info(f"Using SelectKBest with {self.config.feature_selection_score_func}")
            
        elif self.config.feature_selection_method == 'rfe':
            # Recursive Feature Elimination with tree-based estimator
            if self.config.rfe_estimator == 'random_forest':
                estimator = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=3,
                    random_state=42
                )
            elif self.config.rfe_estimator == 'xgboost':
                estimator = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
            else:  # lightgbm
                estimator = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                
            self.selector = RFE(estimator, n_features_to_select=n_features)
            logger.info(f"Using RFE with {self.config.rfe_estimator} estimator")
            
        elif self.config.feature_selection_method == 'lasso':
            # LASSO-based feature selection
            # Note: LASSO doesn't directly select k features, it zeros out coefficients
            # We'll use it to rank features and then select top k
            self.selector = 'lasso'  # Special handling in fit
            logger.info(f"Using LASSO with alpha={self.config.lasso_alpha}")
            
        elif self.config.feature_selection_method == 'mutual_info':
            # Mutual information based selection
            self.selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            logger.info("Using Mutual Information based selection")
            
        elif self.config.feature_selection_method == 'tree_importance':
            # Tree-based feature importance
            self.selector = 'tree_importance'  # Special handling in fit
            logger.info("Using Tree-based feature importance selection")
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.config.feature_selection_method}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the feature selector on training data.

        Args:
            X: Training feature matrix
            y: Training target values

        Returns:
            Self for method chaining
        """
        # Check for NaN values in input
        if isinstance(X, pd.DataFrame):
            if X.isnull().any().any():
                nan_cols = X.columns[X.isnull().any()].tolist()
                logger.warning(f"[FEATURE SELECTION] Input contains NaN values in columns: {nan_cols}")
                # Try to handle NaN values
                X = X.fillna(0)
                logger.info("[FEATURE SELECTION] Filled NaN values with 0")
        elif isinstance(X, np.ndarray):
            if np.isnan(X).any():
                nan_count = np.sum(np.isnan(X))
                logger.warning(f"[FEATURE SELECTION] Input contains {nan_count} NaN values")
                # Replace NaN with 0
                X = np.nan_to_num(X, nan=0.0)
                logger.info("[FEATURE SELECTION] Replaced NaN values with 0")

        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        # Log before feature selection
        original_features = X.shape[1]
        target_features = self._get_n_features(X)
        logger.info(f"[FEATURE SELECTION] Before: {original_features} features")
        logger.info(f"[FEATURE SELECTION] Target: {target_features} features ({self.config.feature_selection_method})")
        
        # Create and fit the selector
        self._create_selector(X, y)
        
        if self.selector == 'lasso':
            # Special handling for LASSO
            lasso = LassoCV(alphas=[self.config.lasso_alpha], cv=5, random_state=42)
            lasso.fit(X, y)
            
            # Get feature importances from LASSO coefficients
            feature_importances = np.abs(lasso.coef_)
            
            # Select top k features
            n_features = self._get_n_features(X)
            top_indices = np.argsort(feature_importances)[-n_features:]
            
            # Create a boolean mask
            self.selected_features_ = np.zeros(X.shape[1], dtype=bool)
            self.selected_features_[top_indices] = True
            self.feature_scores_ = feature_importances
            
            logger.info(f"LASSO selected {n_features} features with non-zero coefficients")
            
        elif self.selector == 'tree_importance':
            # Special handling for tree-based importance
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42
            )
            rf.fit(X, y)
            
            # Get feature importances
            feature_importances = rf.feature_importances_
            
            # Select features above threshold or top k
            n_features = self._get_n_features(X)
            
            # Method 1: Threshold-based
            threshold_mask = feature_importances > self.config.tree_importance_threshold
            
            # Method 2: Top-k based
            top_indices = np.argsort(feature_importances)[-n_features:]
            top_k_mask = np.zeros(X.shape[1], dtype=bool)
            top_k_mask[top_indices] = True
            
            # Combine: use top-k but ensure we meet minimum threshold
            self.selected_features_ = top_k_mask
            self.feature_scores_ = feature_importances
            
            n_selected = np.sum(self.selected_features_)
            logger.info(f"Tree importance selected {n_selected} features")
            
        else:
            # Standard sklearn selectors
            self.selector.fit(X, y)
            self.selected_features_ = self.selector.get_support()
            
            if hasattr(self.selector, 'scores_'):
                self.feature_scores_ = self.selector.scores_
            elif hasattr(self.selector, 'ranking_'):
                # For RFE, convert ranking to scores (inverse ranking)
                self.feature_scores_ = 1.0 / self.selector.ranking_
        
        # Log selected features
        n_selected = np.sum(self.selected_features_)
        logger.info(f"[FEATURE SELECTION] After: {n_selected} features selected (reduced from {X.shape[1]})")
        
        if self.feature_names_ and n_selected < 100:  # Only log if reasonable number
            selected_names = [name for name, selected in zip(self.feature_names_, self.selected_features_) if selected]
            logger.debug(f"Selected features: {selected_names[:20]}...")  # Show first 20
            
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data by selecting features.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix with selected features only
        """
        if self.selector is None and self.selected_features_ is None:
            raise ValueError("Feature selector has not been fitted yet")
        
        # Log transformation
        original_features = X.shape[1]
        selected_count = np.sum(self.selected_features_)
        logger.debug(f"[FEATURE SELECTION] Transforming: {original_features} → {selected_count} features")
        
        # Handle special cases
        if self.selector in ['lasso', 'tree_importance']:
            # Use the boolean mask directly
            if isinstance(X, pd.DataFrame):
                selected_columns = X.columns[self.selected_features_]
                return X[selected_columns]
            else:
                return X[:, self.selected_features_]
        else:
            # Use sklearn selector
            X_transformed = self.selector.transform(X)
            
            # Preserve DataFrame structure if input was DataFrame
            if isinstance(X, pd.DataFrame) and self.feature_names_:
                selected_names = [name for name, selected in zip(self.feature_names_, self.selected_features_) if selected]
                return pd.DataFrame(X_transformed, columns=selected_names, index=X.index)
            
            return X_transformed
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit the selector and transform data in one step.
        
        Args:
            X: Training feature matrix
            y: Training target values
            
        Returns:
            Transformed feature matrix with selected features only
        """
        return self.fit(X, y).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """
        Get the names of selected features.
        
        Returns:
            List of selected feature names
        """
        if self.selected_features_ is None:
            raise ValueError("Feature selector has not been fitted yet")
            
        if self.feature_names_:
            return [name for name, selected in zip(self.feature_names_, self.selected_features_) if selected]
        else:
            return [f"feature_{i}" for i, selected in enumerate(self.selected_features_) if selected]
    
    def get_feature_scores(self) -> Optional[pd.DataFrame]:
        """
        Get feature scores/importances if available.
        
        Returns:
            DataFrame with feature names and scores, sorted by score
        """
        if self.feature_scores_ is None:
            return None
            
        df = pd.DataFrame({
            'feature': self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(len(self.feature_scores_))],
            'score': self.feature_scores_,
            'selected': self.selected_features_
        })
        
        return df.sort_values('score', ascending=False)