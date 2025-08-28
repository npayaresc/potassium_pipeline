"""
Advanced Autoencoder Implementations for Spectral Data Dimension Reduction

This module provides sophisticated autoencoder variants specifically designed
for spectral and tabular data dimension reduction.
"""
import logging
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

from .dimension_reduction import DimensionReducer

logger = logging.getLogger(__name__)


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (VAE) for probabilistic dimension reduction."""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_layers: List[int]):
        super().__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters (mean and log variance)
        self.fc_mu = nn.Linear(prev_dim, encoding_dim)
        self.fc_logvar = nn.Linear(prev_dim, encoding_dim)
        
        # Decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder for robust feature learning."""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_layers: List[int], 
                 noise_factor: float = 0.2):
        super().__init__()
        self.noise_factor = noise_factor
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)  # Higher dropout for denoising
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def add_noise(self, x):
        """Add Gaussian noise to input."""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise
    
    def forward(self, x, add_noise=True):
        """Forward pass with optional noise addition."""
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded, encoded


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 regularization for interpretable features."""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_layers: List[int],
                 sparsity_param: float = 0.05, beta: float = 3.0):
        super().__init__()
        self.sparsity_param = sparsity_param
        self.beta = beta
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.Sigmoid())  # Sigmoid for sparsity
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass through sparse autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def kl_divergence(self, rho, rho_hat):
        """Calculate KL divergence for sparsity constraint."""
        return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


class ContractiveAutoencoder(nn.Module):
    """Contractive Autoencoder for learning robust representations."""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_layers: List[int],
                 lambda_contractive: float = 1e-4):
        super().__init__()
        self.lambda_contractive = lambda_contractive
        
        # Build encoder with individual layers for Jacobian computation
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layer = nn.Linear(prev_dim, hidden_dim)
            self.encoder_layers.append(layer)
            prev_dim = hidden_dim
        
        self.encoder_final = nn.Linear(prev_dim, encoding_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode with tracking for Jacobian."""
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h))
        return torch.sigmoid(self.encoder_final(h))
    
    def forward(self, x):
        """Forward pass."""
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def jacobian_norm(self, x, h):
        """Compute Frobenius norm of Jacobian for contractive loss."""
        # Simplified approximation for efficiency
        dh = h * (1 - h)  # Derivative of sigmoid
        w = self.encoder_final.weight
        jacobian = dh.unsqueeze(2) * w.unsqueeze(0)
        return torch.sum(jacobian ** 2, dim=(1, 2)).mean()


class VAEReducer(DimensionReducer):
    """Variational Autoencoder dimension reducer with automatic component selection."""
    
    def __init__(self, n_components: Union[int, str] = 10, hidden_layers: Optional[List[int]] = None,
                 epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                 beta: float = 1.0, device: str = 'auto', auto_components_method: str = 'elbow',
                 auto_components_range: Tuple[int, int] = (5, 20), **kwargs):
        """
        Initialize VAE reducer with automatic component selection.
        
        Args:
            n_components: Dimension of latent space, or 'auto' for automatic selection
            hidden_layers: List of hidden layer sizes
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            beta: Weight for KL divergence term (beta-VAE)
            device: Device to use ('cpu', 'cuda', or 'auto')
            auto_components_method: Method for automatic selection ('elbow', 'reconstruction_threshold')
            auto_components_range: Range to search for optimal components (min, max)
        """
        self.n_components = n_components
        self.hidden_layers = hidden_layers or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.device = device
        self.auto_components_method = auto_components_method
        self.auto_components_range = auto_components_range
        self.kwargs = kwargs
        
        self.model = None
        self.scaler = None
        self.n_features_ = None
        self._device = None
        self.optimal_components_ = None
    
    def _get_device(self):
        """Determine the device to use."""
        if self._device is None:
            if self.device == 'auto':
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self._device = torch.device(self.device)
        return self._device
    
    def _vae_loss(self, recon_x, x, mu, logvar):
        """VAE loss = Reconstruction loss + KL divergence."""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss
    
    def _find_optimal_components(self, X_scaled, device):
        """Find optimal number of components using specified method."""
        min_comp, max_comp = self.auto_components_range
        component_scores = []
        
        logger.info(f"Searching for optimal VAE components in range [{min_comp}, {max_comp}] using {self.auto_components_method} method")
        
        # Test different numbers of components
        for n_comp in range(min_comp, max_comp + 1):
            # Build and train model with n_comp components
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            temp_model = VariationalAutoencoder(
                self.n_features_, n_comp, self.hidden_layers
            ).to(device)
            
            optimizer = optim.Adam(temp_model.parameters(), lr=self.learning_rate)
            
            # Shorter training for component selection
            train_epochs = min(50, self.epochs // 2)
            temp_model.train()
            
            for epoch in range(train_epochs):
                total_loss = 0
                for batch in dataloader:
                    x = batch[0]
                    recon, mu, logvar = temp_model(x)
                    loss = self._vae_loss(recon, x, mu, logvar)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            # Evaluate reconstruction quality
            temp_model.eval()
            total_recon_error = 0
            total_kl_div = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    x = batch[0]
                    recon, mu, logvar = temp_model(x)
                    
                    recon_error = F.mse_loss(recon, x, reduction='sum').item()
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
                    
                    total_recon_error += recon_error
                    total_kl_div += kl_div
            
            avg_recon_error = total_recon_error / len(dataloader.dataset)
            avg_kl_div = total_kl_div / len(dataloader.dataset)
            
            if self.auto_components_method == 'elbow':
                # For elbow method, use reconstruction error
                score = avg_recon_error
            elif self.auto_components_method == 'reconstruction_threshold':
                # Balance reconstruction and regularization
                score = avg_recon_error + 0.1 * avg_kl_div
            else:
                score = avg_recon_error
            
            component_scores.append((n_comp, score))
            logger.info(f"  n_components={n_comp}: reconstruction_error={avg_recon_error:.4f}, score={score:.4f}")
        
        # Find optimal components using chosen method
        if self.auto_components_method == 'elbow':
            optimal_components = self._find_elbow_point(component_scores)
        elif self.auto_components_method == 'reconstruction_threshold':
            # Find point where improvement becomes marginal (< 5% improvement)
            optimal_components = self._find_threshold_point(component_scores, threshold=0.05)
        else:
            # Default: minimum reconstruction error
            optimal_components = min(component_scores, key=lambda x: x[1])[0]
        
        logger.info(f"Selected optimal components: {optimal_components}")
        return optimal_components
    
    def _find_elbow_point(self, scores):
        """Find elbow point in reconstruction error curve."""
        if len(scores) < 3:
            return scores[0][0]
        
        # Calculate second derivatives to find elbow
        components = [s[0] for s in scores]
        errors = [s[1] for s in scores]
        
        # Normalize for comparison
        min_error, max_error = min(errors), max(errors)
        if max_error - min_error == 0:
            return components[0]
        
        normalized_errors = [(e - min_error) / (max_error - min_error) for e in errors]
        
        # Simple elbow detection: find maximum distance from line connecting first and last points
        max_distance = 0
        elbow_idx = 0
        
        for i in range(1, len(components) - 1):
            # Distance from point to line
            x1, y1 = 0, normalized_errors[0]
            x2, y2 = len(components) - 1, normalized_errors[-1]
            x0, y0 = i, normalized_errors[i]
            
            # Point-to-line distance formula
            distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            
            if distance > max_distance:
                max_distance = distance
                elbow_idx = i
        
        return components[elbow_idx]
    
    def _find_threshold_point(self, scores, threshold=0.05):
        """Find point where improvement becomes less than threshold."""
        if len(scores) < 2:
            return scores[0][0]
        
        for i in range(1, len(scores)):
            prev_error = scores[i-1][1]
            curr_error = scores[i][1]
            
            if prev_error == 0:
                continue
                
            improvement = (prev_error - curr_error) / prev_error
            if improvement < threshold:
                return scores[i-1][0]  # Return previous component count
        
        # If no threshold found, return last component count
        return scores[-1][0]

    def fit(self, X, y=None):
        """Fit the VAE with automatic component selection if enabled."""
        device = self._get_device()
        self.n_features_ = X.shape[1]
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of components
        if self.n_components == 'auto':
            self.optimal_components_ = self._find_optimal_components(X_scaled, device)
        else:
            self.optimal_components_ = self.n_components
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build final model with optimal components
        self.model = VariationalAutoencoder(
            self.n_features_, self.optimal_components_, self.hidden_layers
        ).to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Train final model
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                # Forward pass
                recon, mu, logvar = self.model(x)
                loss = self._vae_loss(recon, x, mu, logvar)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader.dataset)
                logger.info(f"VAE epoch {epoch + 1}/{self.epochs}, loss: {avg_loss:.4f}")
        
        self.model.eval()
        logger.info(f"VAE fitted: {self.n_features_} → {self.optimal_components_} components")
        return self
    
    def transform(self, X):
        """Transform using VAE encoder (mean of latent distribution)."""
        if self.model is None:
            raise ValueError("VAE must be fitted before transform")
        
        device = self._get_device()
        self.model.eval()
        
        # Scale and convert
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Encode to latent space (use mean)
        with torch.no_grad():
            mu, _ = self.model.encode(X_tensor)
        
        return mu.cpu().numpy()
    
    def get_n_components(self):
        """Return number of components (optimal if auto-determined)."""
        return self.optimal_components_ if self.optimal_components_ is not None else self.n_components


class DenoisingAutoencoderReducer(DimensionReducer):
    """Denoising Autoencoder dimension reducer."""
    
    def __init__(self, n_components: int = 10, hidden_layers: Optional[List[int]] = None,
                 epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                 noise_factor: float = 0.2, device: str = 'auto', **kwargs):
        """
        Initialize Denoising Autoencoder reducer.
        
        Args:
            n_components: Encoding dimension
            hidden_layers: List of hidden layer sizes
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            noise_factor: Standard deviation of Gaussian noise
            device: Device to use
        """
        self.n_components = n_components
        self.hidden_layers = hidden_layers or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise_factor = noise_factor
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
    
    def fit(self, X, y=None):
        """Fit the denoising autoencoder."""
        device = self._get_device()
        self.n_features_ = X.shape[1]
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build model
        self.model = DenoisingAutoencoder(
            self.n_features_, self.n_components, self.hidden_layers, self.noise_factor
        ).to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                # Forward pass (add noise during training)
                recon, _ = self.model(x, add_noise=True)
                loss = criterion(recon, x)  # Reconstruct clean input
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Denoising AE epoch {epoch + 1}/{self.epochs}, "
                           f"reconstruction loss: {avg_loss:.4f}")
        
        self.model.eval()
        logger.info(f"Denoising AE fitted: {self.n_features_} → {self.n_components} components")
        return self
    
    def transform(self, X):
        """Transform using encoder (without adding noise)."""
        if self.model is None:
            raise ValueError("Denoising AE must be fitted before transform")
        
        device = self._get_device()
        self.model.eval()
        
        # Scale and convert
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Encode without noise
        with torch.no_grad():
            _, encoded = self.model(X_tensor, add_noise=False)
        
        return encoded.cpu().numpy()
    
    def get_n_components(self):
        """Return number of components."""
        return self.n_components


class SparseAutoencoderReducer(DimensionReducer):
    """Sparse Autoencoder dimension reducer for interpretable features."""
    
    def __init__(self, n_components: int = 10, hidden_layers: Optional[List[int]] = None,
                 epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                 sparsity_param: float = 0.05, beta: float = 3.0, device: str = 'auto', **kwargs):
        """
        Initialize Sparse Autoencoder reducer.
        
        Args:
            n_components: Encoding dimension
            hidden_layers: List of hidden layer sizes
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            sparsity_param: Target average activation (rho)
            beta: Weight for sparsity penalty
            device: Device to use
        """
        self.n_components = n_components
        self.hidden_layers = hidden_layers or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sparsity_param = sparsity_param
        self.beta = beta
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
    
    def fit(self, X, y=None):
        """Fit the sparse autoencoder."""
        device = self._get_device()
        self.n_features_ = X.shape[1]
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build model
        self.model = SparseAutoencoder(
            self.n_features_, self.n_components, self.hidden_layers,
            self.sparsity_param, self.beta
        ).to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            total_sparsity_loss = 0
            
            for batch in dataloader:
                x = batch[0]
                
                # Forward pass
                recon, encoded = self.model(x)
                
                # Reconstruction loss
                recon_loss = criterion(recon, x)
                
                # Sparsity loss (KL divergence)
                rho_hat = torch.mean(encoded, dim=0)
                sparsity_loss = torch.sum(
                    self.model.kl_divergence(self.sparsity_param, rho_hat)
                )
                
                # Total loss
                loss = recon_loss + self.beta * sparsity_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += recon_loss.item()
                total_sparsity_loss += sparsity_loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                avg_sparsity = total_sparsity_loss / len(dataloader)
                logger.info(f"Sparse AE epoch {epoch + 1}/{self.epochs}, "
                           f"recon loss: {avg_loss:.4f}, sparsity: {avg_sparsity:.4f}")
        
        self.model.eval()
        logger.info(f"Sparse AE fitted: {self.n_features_} → {self.n_components} components")
        return self
    
    def transform(self, X):
        """Transform using encoder."""
        if self.model is None:
            raise ValueError("Sparse AE must be fitted before transform")
        
        device = self._get_device()
        self.model.eval()
        
        # Scale and convert
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Encode
        with torch.no_grad():
            _, encoded = self.model(X_tensor)
        
        return encoded.cpu().numpy()
    
    def get_n_components(self):
        """Return number of components."""
        return self.n_components


# Register the new reducers with the factory
def register_advanced_autoencoders():
    """Register advanced autoencoder types with the dimension reduction factory."""
    from .dimension_reduction import DimensionReductionFactory
    
    DimensionReductionFactory.register_reducer('vae', VAEReducer)
    DimensionReductionFactory.register_reducer('denoising_ae', DenoisingAutoencoderReducer)
    DimensionReductionFactory.register_reducer('sparse_ae', SparseAutoencoderReducer)
    
    logger.info("Registered advanced autoencoder reducers: vae, denoising_ae, sparse_ae")


# Spectral-specific autoencoder with 1D convolutions
class SpectralAutoencoder(nn.Module):
    """1D Convolutional Autoencoder specifically designed for spectral data."""
    
    def __init__(self, input_dim: int, encoding_dim: int, conv_channels: List[int] = [32, 64, 128]):
        super().__init__()
        
        # Calculate conv output size
        conv_out_size = input_dim
        for _ in conv_channels:
            conv_out_size = (conv_out_size - 2) // 2  # kernel=3, stride=2, padding=0
        
        # Encoder: 1D Convolutions
        encoder_conv = []
        in_channels = 1
        
        for out_channels in conv_channels:
            encoder_conv.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout1d(0.2)
            ])
            in_channels = out_channels
        
        self.encoder_conv = nn.Sequential(*encoder_conv)
        self.encoder_fc = nn.Linear(conv_channels[-1] * conv_out_size, encoding_dim)
        
        # Decoder: Fully connected + 1D Transposed Convolutions
        self.decoder_fc = nn.Linear(encoding_dim, conv_channels[-1] * conv_out_size)
        
        decoder_conv = []
        conv_channels_reversed = list(reversed(conv_channels))
        
        for i in range(len(conv_channels_reversed) - 1):
            decoder_conv.extend([
                nn.ConvTranspose1d(conv_channels_reversed[i], conv_channels_reversed[i+1],
                                   kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm1d(conv_channels_reversed[i+1]),
                nn.Dropout1d(0.2)
            ])
        
        # Final layer to reconstruct original dimension
        decoder_conv.append(
            nn.ConvTranspose1d(conv_channels_reversed[-1], 1, kernel_size=3, stride=2, padding=0)
        )
        
        self.decoder_conv = nn.Sequential(*decoder_conv)
        self.conv_out_size = conv_out_size
        self.conv_channels = conv_channels
    
    def forward(self, x):
        """Forward pass through spectral autoencoder."""
        # Add channel dimension for conv1d
        x = x.unsqueeze(1)  # [batch, 1, length]
        
        # Encode
        conv_out = self.encoder_conv(x)
        conv_flat = conv_out.view(conv_out.size(0), -1)
        encoded = self.encoder_fc(conv_flat)
        
        # Decode
        fc_out = self.decoder_fc(encoded)
        fc_reshaped = fc_out.view(-1, self.conv_channels[-1], self.conv_out_size)
        decoded = self.decoder_conv(fc_reshaped)
        
        # Remove channel dimension
        decoded = decoded.squeeze(1)
        
        return decoded, encoded