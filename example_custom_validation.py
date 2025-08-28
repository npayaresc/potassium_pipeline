#!/usr/bin/env python3
"""
Example script demonstrating how to use custom validation sets in the nitrogen pipeline.

This script shows how to:
1. Organize validation data in a directory structure
2. Run training with a custom validation directory
"""

import shutil
from pathlib import Path

def setup_example_validation_directory():
    """
    Shows how to set up a validation directory with raw spectral files.
    In practice, you would copy your actual validation files here.
    """
    
    # Example directory structure
    validation_dir = Path("data/validation_raw_files")
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created validation directory: {validation_dir}")
    print("\nTo use this feature, copy your raw validation spectral files (.csv.txt) into this directory.")
    print("The pipeline will process them through the same steps as training data:")
    print("  1. Averaging multiple measurements per sample")
    print("  2. Outlier detection and cleansing")
    print("  3. Feature extraction")
    
    return validation_dir

def print_usage_examples(validation_dir):
    """Prints example commands for using the custom validation directory."""
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    
    print("\n1. Train standard models with custom validation directory:")
    print(f"   python main.py train --validation-dir {validation_dir}")
    
    print("\n2. Train with GPU and custom validation directory:")
    print(f"   python main.py train --gpu --validation-dir {validation_dir}")
    
    print("\n3. Run AutoGluon with custom validation directory:")
    print(f"   python main.py autogluon --validation-dir {validation_dir}")
    
    print("\n4. Run hyperparameter tuning with custom validation directory:")
    print(f"   python main.py tune --validation-dir {validation_dir}")
    
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE EXAMPLE:")
    print("="*60)
    print(f"{validation_dir}/")
    print("├── 1053789KENP1S001_G_2025_01_04_1.csv.txt")
    print("├── 1053789KENP1S001_G_2025_01_04_2.csv.txt")
    print("├── 1053789KENP1S001_G_2025_01_04_3.csv.txt")
    print("├── 1053789KENP2S027_G_2025_02_08_1.csv.txt")
    print("├── 1053789KENP2S027_G_2025_02_08_2.csv.txt")
    print("└── ... (more raw spectral files)")
    
    print("\n" + "="*60)
    print("NOTES:")
    print("="*60)
    print("- The validation directory should contain raw spectral files (.csv.txt)")
    print("- Files will be grouped by sample ID and averaged (e.g., _1, _2, _3 files)")
    print("- Only samples that exist in the reference Excel file will be used")
    print("- The same outlier detection and cleansing will be applied")
    print("- If validation samples overlap with training data, they'll be removed from training")
    print("")

def copy_example_files(source_dir, validation_dir, sample_prefixes):
    """
    Example function to copy specific samples to validation directory.
    This is just for demonstration - in practice you'd select your validation files manually.
    """
    source_path = Path(source_dir)
    val_path = Path(validation_dir)
    
    if not source_path.exists():
        print(f"\nSource directory not found: {source_path}")
        print("In practice, you would manually copy your validation files.")
        return
    
    copied_count = 0
    for prefix in sample_prefixes:
        for file in source_path.glob(f"{prefix}*.csv.txt"):
            dest = val_path / file.name
            shutil.copy2(file, dest)
            copied_count += 1
    
    print(f"\nCopied {copied_count} files to validation directory")

if __name__ == "__main__":
    print("Setting up example validation directory structure...")
    validation_dir = setup_example_validation_directory()
    
    # Show usage examples
    print_usage_examples(validation_dir)
    
    # Example of how you might copy files (uncomment and modify as needed)
    # sample_prefixes_for_validation = [
    #     "1053789KENP1S026_G_2025_02_08",
    #     "1053789KENP2S027_G_2025_02_08",
    #     "1053789KENP3S028_G_2025_02_08",
    # ]
    # copy_example_files(
    #     "data/raw/data_5278_Phase3",
    #     validation_dir,
    #     sample_prefixes_for_validation
    # )
    
    print("\nIMPORTANT: Remember to copy your actual validation raw files to the directory!")