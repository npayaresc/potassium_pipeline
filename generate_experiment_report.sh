#!/bin/bash

# Experiment Results Analysis and Reporting Script
# Generates comprehensive reports from experiment results

set -e

# Configuration
RESULTS_DIR="${RESULTS_DIR:-./experiment_results}"
LOGS_DIR="${LOGS_DIR:-./experiment_logs}"
CONFIGS_DIR="${CONFIGS_DIR:-./experiment_configs}"
REPORTS_DIR="${REPORTS_DIR:-./experiment_reports}"
GCS_BUCKET="${BUCKET_NAME:-mapana-ai-models-magnesium-data}"

mkdir -p "$REPORTS_DIR"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

# ============================================================================
# DOWNLOAD RESULTS FROM GCS
# ============================================================================

download_results() {
    log_info "Downloading results from GCS..."
    
    # Create local directories
    mkdir -p "$RESULTS_DIR/gcs_results"
    mkdir -p "$RESULTS_DIR/gcs_models"
    mkdir -p "$RESULTS_DIR/gcs_reports"
    
    # Download training summaries
    gsutil -m cp -r "gs://${GCS_BUCKET}/magnesium-pipeline/reports/*.csv" \
        "$RESULTS_DIR/gcs_reports/" 2>/dev/null || true
    
    # Download model metadata
    gsutil -m cp -r "gs://${GCS_BUCKET}/magnesium-pipeline/models/*.json" \
        "$RESULTS_DIR/gcs_models/" 2>/dev/null || true
    
    log_info "Download complete"
}

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

analyze_results() {
    log_info "Analyzing experiment results..."
    
    # Create comprehensive results CSV
    local summary_file="$REPORTS_DIR/experiment_summary_$(date +%Y%m%d_%H%M%S).csv"
    
    echo "experiment_id,strategy,models,r2_score,mape,mae,rmse,training_time,config_file" > "$summary_file"
    
    # Parse all result files
    for result_file in "$RESULTS_DIR"/gcs_reports/training_summary_*.csv; do
        if [ -f "$result_file" ]; then
            # Extract metrics from CSV (skip header)
            tail -n +2 "$result_file" | while IFS=',' read -r model strategy r2 mape mae rmse time; do
                exp_id=$(basename "$result_file" .csv)
                echo "${exp_id},${strategy},${model},${r2},${mape},${mae},${rmse},${time}," >> "$summary_file"
            done
        fi
    done
    
    log_info "Results analysis saved to: $summary_file"
}

# ============================================================================
# GENERATE PYTHON ANALYSIS SCRIPT
# ============================================================================

create_python_analyzer() {
    cat > "$REPORTS_DIR/analyze_experiments.py" << 'EOF'
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# Configuration
REPORTS_DIR = Path("./experiment_reports")
RESULTS_DIR = Path("./experiment_results")

def load_experiment_data():
    """Load all experiment results"""
    summary_files = list(REPORTS_DIR.glob("experiment_summary_*.csv"))
    if not summary_files:
        print("No summary files found!")
        return None
    
    # Load most recent summary
    latest_summary = sorted(summary_files)[-1]
    df = pd.read_csv(latest_summary)
    
    # Clean data
    df['r2_score'] = pd.to_numeric(df['r2_score'], errors='coerce')
    df['mape'] = pd.to_numeric(df['mape'], errors='coerce')
    df['mae'] = pd.to_numeric(df['mae'], errors='coerce')
    df['rmse'] = pd.to_numeric(df['rmse'], errors='coerce')
    
    return df

def analyze_by_strategy(df):
    """Analyze results by feature strategy"""
    print("\n" + "="*80)
    print("ANALYSIS BY FEATURE STRATEGY")
    print("="*80)
    
    strategies = df.groupby('strategy').agg({
        'r2_score': ['mean', 'max', 'std'],
        'mape': ['mean', 'min', 'std'],
        'mae': ['mean', 'min', 'std']
    }).round(4)
    
    print(strategies)
    
    # Find best strategy
    best_r2_strategy = df.loc[df['r2_score'].idxmax(), 'strategy']
    best_r2_value = df['r2_score'].max()
    print(f"\nBest R² Strategy: {best_r2_strategy} (R² = {best_r2_value:.4f})")

def analyze_by_model(df):
    """Analyze results by model type"""
    print("\n" + "="*80)
    print("ANALYSIS BY MODEL TYPE")
    print("="*80)
    
    models = df.groupby('models').agg({
        'r2_score': ['mean', 'max', 'std'],
        'mape': ['mean', 'min', 'std'],
        'mae': ['mean', 'min', 'std']
    }).round(4)
    
    print(models)
    
    # Find best model
    best_model = df.loc[df['r2_score'].idxmax(), 'models']
    best_r2_value = df['r2_score'].max()
    print(f"\nBest Model: {best_model} (R² = {best_r2_value:.4f})")

def find_target_achieving_experiments(df, r2_target=0.8, mape_target=10, mae_target=0.04):
    """Find experiments that achieve target metrics"""
    print("\n" + "="*80)
    print(f"EXPERIMENTS ACHIEVING TARGETS (R² ≥ {r2_target}, MAPE < {mape_target}%, MAE < {mae_target})")
    print("="*80)
    
    successful = df[
        (df['r2_score'] >= r2_target) &
        (df['mape'] < mape_target) &
        (df['mae'] < mae_target)
    ]
    
    if len(successful) > 0:
        print(f"\n{len(successful)} experiments achieved all targets!")
        print("\nTop 5 by R² score:")
        top5 = successful.nlargest(5, 'r2_score')[['experiment_id', 'strategy', 'models', 'r2_score', 'mape', 'mae']]
        print(top5.to_string(index=False))
    else:
        print("\nNo experiments achieved all targets yet.")
        
        # Show closest attempts
        print("\nClosest attempts (by R² score):")
        closest = df.nlargest(5, 'r2_score')[['experiment_id', 'strategy', 'models', 'r2_score', 'mape', 'mae']]
        print(closest.to_string(index=False))

def generate_visualizations(df):
    """Generate visualization plots"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. R² by Strategy
    ax = axes[0, 0]
    df.boxplot(column='r2_score', by='strategy', ax=ax)
    ax.set_title('R² Score by Strategy')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('R² Score')
    
    # 2. MAPE by Strategy
    ax = axes[0, 1]
    df.boxplot(column='mape', by='strategy', ax=ax)
    ax.set_title('MAPE by Strategy')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('MAPE (%)')
    
    # 3. MAE by Strategy
    ax = axes[0, 2]
    df.boxplot(column='mae', by='strategy', ax=ax)
    ax.set_title('MAE by Strategy')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('MAE')
    
    # 4. R² Distribution
    ax = axes[1, 0]
    df['r2_score'].hist(bins=30, ax=ax)
    ax.axvline(0.8, color='r', linestyle='--', label='Target (0.8)')
    ax.set_title('R² Score Distribution')
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Count')
    ax.legend()
    
    # 5. MAPE vs R²
    ax = axes[1, 1]
    ax.scatter(df['r2_score'], df['mape'], alpha=0.5)
    ax.axvline(0.8, color='r', linestyle='--', alpha=0.5)
    ax.axhline(10, color='r', linestyle='--', alpha=0.5)
    ax.set_title('MAPE vs R² Score')
    ax.set_xlabel('R² Score')
    ax.set_ylabel('MAPE (%)')
    
    # 6. MAE vs R²
    ax = axes[1, 2]
    ax.scatter(df['r2_score'], df['mae'], alpha=0.5)
    ax.axvline(0.8, color='r', linestyle='--', alpha=0.5)
    ax.axhline(0.04, color='r', linestyle='--', alpha=0.5)
    ax.set_title('MAE vs R² Score')
    ax.set_xlabel('R² Score')
    ax.set_ylabel('MAE')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f'experiment_analysis_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
    print(f"Visualizations saved to {REPORTS_DIR}")

def generate_recommendations(df):
    """Generate recommendations based on results"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find best configurations
    best_r2 = df.nlargest(1, 'r2_score').iloc[0]
    best_mape = df.nsmallest(1, 'mape').iloc[0]
    best_mae = df.nsmallest(1, 'mae').iloc[0]
    
    print("\n1. BEST CONFIGURATIONS:")
    print(f"   - Best R²: {best_r2['strategy']} with {best_r2['models']} (R² = {best_r2['r2_score']:.4f})")
    print(f"   - Best MAPE: {best_mape['strategy']} with {best_mape['models']} (MAPE = {best_mape['mape']:.2f}%)")
    print(f"   - Best MAE: {best_mae['strategy']} with {best_mae['models']} (MAE = {best_mae['mae']:.4f})")
    
    # Strategy recommendations
    strategy_stats = df.groupby('strategy')['r2_score'].agg(['mean', 'max', 'count'])
    best_avg_strategy = strategy_stats['mean'].idxmax()
    
    print("\n2. STRATEGY RECOMMENDATIONS:")
    print(f"   - Most consistent: {best_avg_strategy} (avg R² = {strategy_stats.loc[best_avg_strategy, 'mean']:.4f})")
    print(f"   - Most tested: {strategy_stats['count'].idxmax()} ({strategy_stats['count'].max()} experiments)")
    
    # Model recommendations
    model_stats = df.groupby('models')['r2_score'].agg(['mean', 'max', 'count'])
    best_avg_model = model_stats['mean'].idxmax()
    
    print("\n3. MODEL RECOMMENDATIONS:")
    print(f"   - Most consistent: {best_avg_model} (avg R² = {model_stats.loc[best_avg_model, 'mean']:.4f})")
    print(f"   - Best performing: {model_stats['max'].idxmax()} (max R² = {model_stats['max'].max():.4f})")
    
    # Next steps
    print("\n4. NEXT STEPS:")
    if df['r2_score'].max() >= 0.8:
        print("   ✓ Target R² achieved! Focus on reducing MAPE and MAE")
    else:
        gap = 0.8 - df['r2_score'].max()
        print(f"   - Need {gap:.4f} improvement in R² to reach target")
        print("   - Consider: Extended training, more features, or ensemble methods")
    
    if df['mape'].min() < 10:
        print("   ✓ Target MAPE achieved!")
    else:
        print(f"   - Need {df['mape'].min() - 10:.2f}% reduction in MAPE")
    
    if df['mae'].min() < 0.04:
        print("   ✓ Target MAE achieved!")
    else:
        print(f"   - Need {df['mae'].min() - 0.04:.4f} reduction in MAE")

def main():
    """Main analysis function"""
    print("="*80)
    print("MAGNESIUM PIPELINE EXPERIMENT ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_experiment_data()
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} experiment results")
    
    # Run analyses
    analyze_by_strategy(df)
    analyze_by_model(df)
    find_target_achieving_experiments(df)
    generate_visualizations(df)
    generate_recommendations(df)
    
    # Save full report
    report_file = REPORTS_DIR / f"full_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Redirect output to file
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = report_buffer = StringIO()
    
    # Re-run all analyses for report
    analyze_by_strategy(df)
    analyze_by_model(df)
    find_target_achieving_experiments(df)
    generate_recommendations(df)
    
    # Save report
    report_content = report_buffer.getvalue()
    sys.stdout = old_stdout
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"\nFull report saved to: {report_file}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$REPORTS_DIR/analyze_experiments.py"
    log_info "Python analyzer created: $REPORTS_DIR/analyze_experiments.py"
}

# ============================================================================
# GENERATE HTML DASHBOARD
# ============================================================================

generate_html_dashboard() {
    log_info "Generating HTML dashboard..."
    
    local dashboard_file="$REPORTS_DIR/dashboard_$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$dashboard_file" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magnesium Pipeline - Experiment Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .success {
            color: #4caf50;
            font-weight: bold;
        }
        .warning {
            color: #ff9800;
            font-weight: bold;
        }
        .error {
            color: #f44336;
            font-weight: bold;
        }
        .timestamp {
            text-align: right;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Magnesium Pipeline - Experiment Dashboard</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Experiments</div>
                <div class="stat-value" id="total-experiments">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best R² Score</div>
                <div class="stat-value" id="best-r2">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best MAPE</div>
                <div class="stat-value" id="best-mape">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best MAE</div>
                <div class="stat-value" id="best-mae">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Target Achieved</div>
                <div class="stat-value" id="target-achieved">-</div>
            </div>
        </div>
        
        <h2>📊 Experiment Results</h2>
        <div id="results-table">
            <p>Loading results...</p>
        </div>
        
        <h2>🎯 Target Achievement Status</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Target</th>
                <th>Best Achieved</th>
                <th>Status</th>
                <th>Gap</th>
            </tr>
            <tr>
                <td>R² Score</td>
                <td>≥ 0.80</td>
                <td id="r2-achieved">-</td>
                <td id="r2-status">-</td>
                <td id="r2-gap">-</td>
            </tr>
            <tr>
                <td>MAPE</td>
                <td>< 10%</td>
                <td id="mape-achieved">-</td>
                <td id="mape-status">-</td>
                <td id="mape-gap">-</td>
            </tr>
            <tr>
                <td>MAE</td>
                <td>< 0.04</td>
                <td id="mae-achieved">-</td>
                <td id="mae-status">-</td>
                <td id="mae-gap">-</td>
            </tr>
        </table>
        
        <div class="timestamp">
            Generated: <span id="timestamp"></span>
        </div>
    </div>
    
    <script>
        // Update timestamp
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
        
        // Placeholder for dynamic data loading
        // In production, this would load from actual results
        document.getElementById('total-experiments').textContent = 'Loading...';
        document.getElementById('best-r2').textContent = 'Loading...';
        document.getElementById('best-mape').textContent = 'Loading...';
        document.getElementById('best-mae').textContent = 'Loading...';
    </script>
</body>
</html>
EOF
    
    log_info "HTML dashboard generated: $dashboard_file"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "Starting Experiment Report Generation"
    
    # Download latest results from GCS
    download_results
    
    # Analyze results
    analyze_results
    
    # Create Python analyzer
    create_python_analyzer
    
    # Run Python analysis
    if command -v python3 &> /dev/null; then
        log_info "Running Python analysis..."
        cd "$REPORTS_DIR" && python3 analyze_experiments.py
    else
        log_info "Python not available, skipping detailed analysis"
    fi
    
    # Generate HTML dashboard
    generate_html_dashboard
    
    # Generate summary
    log_info "Report generation complete!"
    log_info "Reports available in: $REPORTS_DIR"
    
    # Show summary statistics
    echo ""
    echo "===== QUICK SUMMARY ====="
    if [ -f "$RESULTS_DIR/experiment_tracker.csv" ]; then
        total_exp=$(tail -n +2 "$RESULTS_DIR/experiment_tracker.csv" | wc -l)
        successful=$(grep ",SUCCESS," "$RESULTS_DIR/experiment_tracker.csv" | wc -l)
        failed=$(grep ",FAILED," "$RESULTS_DIR/experiment_tracker.csv" | wc -l)
        
        echo "Total Experiments: $total_exp"
        echo "Successful: $successful"
        echo "Failed: $failed"
        echo "Success Rate: $(( successful * 100 / total_exp ))%"
    fi
    echo "========================="
}

# Run main function
main "$@"