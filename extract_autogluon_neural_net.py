#!/usr/bin/env python3
"""
Extract NeuralNetTorch model from AutoGluon ensemble.

This script analyzes an AutoGluon ensemble to identify the NeuralNetTorch model
and extracts its configuration and parameters for independent optimization.
"""
import logging
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.helpers import setup_logging

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("AutoGluon not available. Please install it with: pip install autogluon")

logger = logging.getLogger(__name__)

def find_autogluon_models():
    """Find all AutoGluon model directories."""
    models_dir = Path("models/autogluon")
    if not models_dir.exists():
        logger.error("No AutoGluon models directory found at models/autogluon")
        return []
    
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(model_dirs)} AutoGluon model directories:")
    for i, model_dir in enumerate(model_dirs, 1):
        # Check if it's a valid AutoGluon model
        predictor_file = model_dir / "predictor.pkl"
        if predictor_file.exists():
            logger.info(f"  {i}. {model_dir.name} ✓")
        else:
            logger.info(f"  {i}. {model_dir.name} (invalid - no predictor.pkl)")
    
    return model_dirs

def analyze_autogluon_ensemble(model_path: Path):
    """
    Analyze AutoGluon ensemble and extract NeuralNetTorch information.
    
    Args:
        model_path: Path to AutoGluon model directory
        
    Returns:
        Dictionary with extracted information
    """
    if not AUTOGLUON_AVAILABLE:
        raise ImportError("AutoGluon is required for model extraction")
    
    logger.info(f"Loading AutoGluon model from: {model_path}")
    predictor = TabularPredictor.load(str(model_path))
    
    # Get leaderboard to see all models
    leaderboard = predictor.leaderboard(silent=True)
    logger.info("\nAutoGluon Model Leaderboard:")
    logger.info("-" * 60)
    for idx, row in leaderboard.iterrows():
        model_name = row['model']
        score = row['score_val']
        pred_time = row.get('pred_time_val', 0)
        fit_time = row.get('fit_time', 0)
        logger.info(f"  {model_name:30s} | Score: {score:8.4f} | Pred: {pred_time:.3f}s | Fit: {fit_time:.1f}s")
    
    # Find NeuralNetTorch models
    neural_net_models = [model for model in leaderboard['model'] if 'NeuralNet' in model]
    
    if not neural_net_models:
        logger.error("No NeuralNetTorch models found in AutoGluon ensemble")
        return None
    
    logger.info(f"\nFound {len(neural_net_models)} NeuralNet models:")
    neural_net_info = {}
    
    for nn_model in neural_net_models:
        row = leaderboard[leaderboard['model'] == nn_model].iloc[0]
        neural_net_info[nn_model] = {
            'score': float(row['score_val']),
            'pred_time': float(row.get('pred_time_val', 0)),
            'fit_time': float(row.get('fit_time', 0))
        }
        logger.info(f"  • {nn_model}: score={row['score_val']:.4f}")
    
    # Get ensemble weights if available
    ensemble_weights = {}
    try:
        if hasattr(predictor, '_trainer'):
            # Try to get the weighted ensemble model
            weighted_models = [m for m in leaderboard['model'] if 'WeightedEnsemble' in m]
            if weighted_models:
                best_ensemble = weighted_models[0]  # Usually the best one
                try:
                    ensemble_model = predictor._trainer.load_model(best_ensemble)
                    if hasattr(ensemble_model, 'model_base'):
                        weights = ensemble_model.model_base.weights_
                        logger.info(f"\nEnsemble Weights from {best_ensemble}:")
                        for model_name, weight in weights.items():
                            if weight > 0:
                                logger.info(f"  • {model_name}: {weight:.3f} ({weight*100:.1f}%)")
                                if 'NeuralNet' in model_name:
                                    ensemble_weights[model_name] = weight
                except Exception as e:
                    logger.debug(f"Could not extract ensemble weights: {e}")
    except Exception as e:
        logger.debug(f"Could not analyze ensemble: {e}")
    
    # Select best NeuralNet model
    best_neural_net = neural_net_models[0]  # First one is typically best
    logger.info(f"\nBest NeuralNet model selected: {best_neural_net}")
    
    # Try to get model configuration
    model_config = {}
    try:
        if hasattr(predictor, '_trainer'):
            # Get model info
            model_info = predictor._trainer.get_model_info(best_neural_net)
            if model_info:
                logger.info("\nExtracted Model Info:")
                for key, value in model_info.items():
                    if key not in ['val_score', 'hyperparameters']:
                        logger.info(f"  {key}: {value}")
                model_config.update(model_info)
            
            # Try to get hyperparameters
            try:
                model_obj = predictor._trainer.load_model(best_neural_net)
                if hasattr(model_obj, 'params'):
                    logger.info("\nModel Hyperparameters:")
                    for key, value in model_obj.params.items():
                        logger.info(f"  {key}: {value}")
                        model_config[f'param_{key}'] = value
            except Exception as e:
                logger.debug(f"Could not load model object: {e}")
                
    except Exception as e:
        logger.debug(f"Could not get model info: {e}")
    
    # Get feature information
    feature_info = {}
    try:
        if hasattr(predictor, 'feature_metadata'):
            features = list(predictor.feature_metadata.get_features())
            feature_info['num_features'] = len(features)
            feature_info['feature_names'] = features[:10]  # First 10 for preview
            logger.info(f"\nModel expects {len(features)} features")
            logger.info(f"First 10 features: {features[:10]}")
    except Exception as e:
        logger.debug(f"Could not get feature metadata: {e}")
    
    # Compile extraction results
    extraction_result = {
        'model_name': best_neural_net,
        'model_type': 'NeuralNetTorch',
        'autogluon_path': str(model_path),
        'extraction_timestamp': pd.Timestamp.now().isoformat(),
        'score': neural_net_info[best_neural_net]['score'],
        'ensemble_weight': ensemble_weights.get(best_neural_net, 0),
        'feature_info': feature_info,
        'model_config': model_config,
        'all_neural_nets': neural_net_info,
        'optimization_ready': True
    }
    
    return extraction_result

def save_extraction_results(extraction_result: dict, output_dir: Path):
    """
    Save extraction results to JSON file.
    
    Args:
        extraction_result: Dictionary with extraction results
        output_dir: Directory to save results
        
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"neural_net_extraction_{timestamp}.json"
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(extraction_result, f, indent=2, default=str)
    
    logger.info(f"\nExtraction results saved to: {output_file}")
    
    # Also create a simplified config for the optimizer
    optimizer_config = {
        'model_source': extraction_result['model_name'],
        'autogluon_score': extraction_result['score'],
        'ensemble_weight': extraction_result['ensemble_weight'],
        'num_features': extraction_result['feature_info'].get('num_features', 'unknown'),
        'extraction_file': str(output_file),
        'suggested_hyperparameters': {
            'hidden_layers': [
                [256, 128, 64, 32],
                [128, 64, 32, 16],
                [512, 256, 128, 64],
                [64, 32, 16]
            ],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'weight_decay': [0, 1e-6, 1e-5, 1e-4, 1e-3],
            'dropout_prob': [0.1, 0.2, 0.3, 0.4, 0.5],
            'batch_size': [16, 32, 64, 128],
            'epochs': [50, 100, 150, 200, 300],
            'early_stopping_patience': [10, 15, 20, 25, 30]
        }
    }
    
    optimizer_config_file = output_dir / f"optimizer_config_{timestamp}.json"
    with open(optimizer_config_file, 'w') as f:
        json.dump(optimizer_config, f, indent=2)
    
    logger.info(f"Optimizer config saved to: {optimizer_config_file}")
    
    return output_file, optimizer_config_file

def main():
    parser = argparse.ArgumentParser(
        description="Extract NeuralNetTorch from AutoGluon ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available AutoGluon models
  python extract_autogluon_neural_net.py --list-models
  
  # Extract from specific model
  python extract_autogluon_neural_net.py --model-path models/autogluon/simple_only_20250122_143052
  
  # Extract and save to custom directory
  python extract_autogluon_neural_net.py --model-path models/autogluon/simple_only_20250122_143052 --output-dir extracted_models
        """
    )
    parser.add_argument("--model-path", type=str, help="Path to AutoGluon model directory")
    parser.add_argument("--list-models", action="store_true", help="List available AutoGluon models")
    parser.add_argument("--output-dir", type=str, default="models/extracted_neural_nets", 
                       help="Output directory for extraction results (default: models/extracted_neural_nets)")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if not AUTOGLUON_AVAILABLE:
        logger.error("AutoGluon is not installed. Please install it with: pip install autogluon")
        sys.exit(1)
    
    if args.list_models:
        model_dirs = find_autogluon_models()
        if model_dirs:
            print("\nTo extract from a model, use:")
            for model_dir in model_dirs:
                print(f"  python {Path(__file__).name} --model-path {model_dir}")
        return
    
    if not args.model_path:
        logger.error("Please specify --model-path or use --list-models to see available models")
        parser.print_help()
        sys.exit(1)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)
    
    if not (model_path / "predictor.pkl").exists():
        logger.error(f"Not a valid AutoGluon model directory (no predictor.pkl): {model_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Extract neural network information
    logger.info("=" * 60)
    logger.info("Starting NeuralNetTorch Extraction")
    logger.info("=" * 60)
    
    extraction_result = analyze_autogluon_ensemble(model_path)
    
    if extraction_result:
        # Save extraction results
        extraction_file, config_file = save_extraction_results(extraction_result, output_dir)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 EXTRACTION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"✓ Extracted: {extraction_result['model_name']}")
        logger.info(f"✓ Score: {extraction_result['score']:.4f}")
        if extraction_result['ensemble_weight'] > 0:
            logger.info(f"✓ Ensemble Weight: {extraction_result['ensemble_weight']*100:.1f}%")
        logger.info(f"✓ Results saved to: {extraction_file}")
        logger.info(f"✓ Optimizer config: {config_file}")
        
        logger.info("\n📝 Next Steps:")
        logger.info("1. Review the extraction results in the JSON file")
        logger.info("2. Run the optimizer with the extracted configuration:")
        logger.info(f"   python optimize_extracted_neural_net.py --config {config_file}")
        logger.info("\nThe optimizer will create a neural network with similar architecture")
        logger.info("and optimize it specifically for your dataset.")
        
    else:
        logger.error("Failed to extract neural network model")
        sys.exit(1)

if __name__ == "__main__":
    main()