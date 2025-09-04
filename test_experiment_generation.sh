#!/bin/bash

# Test script to debug experiment generation

test_experiment_generation() {
    local strategies=("simple_only" "full_context")
    local ag_configs=("best_quality_stacked" "best_quality_long")
    local feature_configs=("mg_focused" "mg_comprehensive")
    local concentration_configs=("weights_improved" "weights_legacy")
    local dimension_configs=("no_reduction" "pls_optimal")
    local gpu_configs=("true" "false")

    local experiments=()
    
    echo "Starting experiment generation..."
    
    # Generate comprehensive experiment matrix
    for strategy in "${strategies[@]}"; do
        echo "Processing strategy: $strategy"
        for ag_config in "${ag_configs[@]}"; do
            echo "  Processing ag_config: $ag_config"
            for feature_config in "${feature_configs[@]}"; do
                echo "    Processing feature_config: $feature_config"
                for concentration_config in "${concentration_configs[@]}"; do
                    echo "      Processing concentration_config: $concentration_config"
                    for dimension_config in "${dimension_configs[@]}"; do
                        echo "        Processing dimension_config: $dimension_config"
                        for gpu in "${gpu_configs[@]}"; do
                            local time_limit="14400"
                            local machine_type="n1-highmem-16"
                            local description="$strategy strategy, $ag_config preset"
                            
                            echo "          Adding: $strategy|$ag_config|$time_limit|$feature_config|$concentration_config|$dimension_config|$gpu|$machine_type|$description"
                            experiments+=("$strategy|$ag_config|$time_limit|$feature_config|$concentration_config|$dimension_config|$gpu|$machine_type|$description")
                        done
                    done
                done
            done
        done
    done
    
    echo "Total experiments generated: ${#experiments[@]}"
    echo "First 5 experiments:"
    for i in {0..4}; do
        if [ $i -lt ${#experiments[@]} ]; then
            echo "$i: ${experiments[$i]}"
        fi
    done
}

test_experiment_generation