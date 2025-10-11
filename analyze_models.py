#!/usr/bin/env python3
"""Analyze models directory structure and features."""
import os
from pathlib import Path

models_dir = Path("/home/payanico/potassium_pipeline/src/models")

print("=" * 80)
print("MODELS DIRECTORY ANALYSIS")
print("=" * 80)
print()

# Categories of files
optimizers = []
trainers = []
utils = []
other = []

for file in sorted(models_dir.glob("*.py")):
    if file.name == "__init__.py":
        continue
        
    with open(file, 'r') as f:
        content = f.read()
        lines = len(content.splitlines())
        
    # Check for key features
    has_feature_selection = "feature_selection" in content
    has_pipeline_steps = "Build pipeline steps" in content
    has_gpu_support = "use_gpu" in content or "GPU" in content
    has_parallel = "parallel" in content.lower()
    
    info = {
        'name': file.name,
        'lines': lines,
        'feature_selection': has_feature_selection,
        'pipeline_steps': has_pipeline_steps,
        'gpu_support': has_gpu_support,
        'parallel': has_parallel
    }
    
    # Categorize
    if file.name.startswith("optimize_"):
        optimizers.append(info)
    elif "trainer" in file.name:
        trainers.append(info)
    elif any(x in file.name for x in ["utils", "helper", "base", "wrapper", "objectives"]):
        utils.append(info)
    else:
        other.append(info)

# Print results
def print_category(name, files):
    print(f"\n{name} ({len(files)} files)")
    print("-" * 80)
    print(f"{'File':<40} {'Lines':>8} {'FS':^5} {'PS':^5} {'GPU':^5} {'PAR':^5}")
    print("-" * 80)
    for f in files:
        print(f"{f['name']:<40} {f['lines']:>8} "
              f"{'✓' if f['feature_selection'] else '✗':^5} "
              f"{'✓' if f['pipeline_steps'] else '✗':^5} "
              f"{'✓' if f['gpu_support'] else '✗':^5} "
              f"{'✓' if f['parallel'] else '✗':^5}")

print_category("OPTIMIZERS", optimizers)
print_category("TRAINERS", trainers)
print_category("UTILITIES", utils)
print_category("OTHER", other)

print("\n" + "=" * 80)
print("LEGEND: FS=Feature Selection, PS=Pipeline Steps, GPU=GPU Support, PAR=Parallel")
print("=" * 80)

# Summary of issues
print("\n🔍 ISSUES DETECTED:")
print("-" * 40)

missing_fs = [f['name'] for f in optimizers if f['pipeline_steps'] and not f['feature_selection']]
if missing_fs:
    print(f"⚠️  Optimizers missing feature selection: {', '.join(missing_fs)}")

print("\n📊 STATISTICS:")
print("-" * 40)
total_lines = sum(f['lines'] for f in optimizers + trainers + utils + other)
print(f"Total lines of code: {total_lines:,}")
print(f"Average file size: {total_lines // (len(optimizers) + len(trainers) + len(utils) + len(other)):,} lines")
print(f"Optimizers with GPU support: {sum(1 for f in optimizers if f['gpu_support'])}/{len(optimizers)}")
print(f"Files with parallel support: {sum(1 for f in optimizers + trainers if f['parallel'])}/{len(optimizers) + len(trainers)}")