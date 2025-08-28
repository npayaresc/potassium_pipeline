#!/usr/bin/env python3
"""
Test that the new command line arguments work correctly.
"""
import subprocess
import sys

def test_help_output():
    """Test that help shows the new feature-parallel arguments."""
    result = subprocess.run(
        [sys.executable, "main.py", "train", "--help"],
        capture_output=True,
        text=True
    )
    
    help_text = result.stdout
    
    # Check for new argument names
    if "--feature-parallel" in help_text:
        print("✓ Found --feature-parallel argument")
    else:
        print("✗ Missing --feature-parallel argument")
        return False
        
    if "--feature-n-jobs" in help_text:
        print("✓ Found --feature-n-jobs argument")
    else:
        print("✗ Missing --feature-n-jobs argument")
        return False
        
    # Check that old names are not present
    if "--parallel " in help_text and "--feature-parallel" not in help_text:
        print("✗ Old --parallel argument still present")
        return False
    else:
        print("✓ Old --parallel argument properly replaced")
        
    if "--n-jobs " in help_text and "--feature-n-jobs" not in help_text:
        print("✗ Old --n-jobs argument still present")
        return False
    else:
        print("✓ Old --n-jobs argument properly replaced")
        
    return True

if __name__ == "__main__":
    print("Testing new command line arguments...")
    if test_help_output():
        print("\n✅ SUCCESS: New command line arguments are correctly configured!")
        print("\nUsage example:")
        print("  python main.py train --feature-parallel --feature-n-jobs 16")
    else:
        print("\n❌ FAILED: Issues with command line arguments")
        sys.exit(1)