#!/usr/bin/env python3
"""Test script to verify external validation directory functionality for tuning."""
import subprocess
import sys

def test_tune_with_validation():
    """Test the tune command with external validation directory."""
    
    # Example validation directory path - adjust as needed
    validation_dir = "/home/payanico/nitrogen_pipeline/data/raw/validation_data"
    
    print("Testing tune command with external validation directory...")
    print(f"Validation directory: {validation_dir}")
    print("-" * 80)
    
    # Build the command
    cmd = [
        sys.executable,
        "main.py",
        "tune",
        "--validation-dir", validation_dir,
        "--gpu"  # Optional: remove if GPU not needed
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        
        # Check return code
        if result.returncode == 0:
            print("\n✓ Command executed successfully!")
        else:
            print(f"\n✗ Command failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n✗ Error running command: {e}")
        
    print("-" * 80)
    print("\nTo run tuning with external validation manually, use:")
    print(f"python main.py tune --validation-dir {validation_dir}")
    print("\nOr without GPU:")
    print(f"python main.py tune --validation-dir {validation_dir}")

if __name__ == "__main__":
    test_tune_with_validation()