#!/bin/bash
# run_all_experiments.sh
# Automated experiment runner for 3 models x 10 runs each

set -e  # Exit on error

# Configuration
MODELS=("runs/roberta-base/best" "runs/microsoft__deberta-v3-large/best" "runs/answerdotai__ModernBERT-large/best")
MODEL_NAMES=("roberta-base" "microsoft__deberta-v3-large" "answerdotai__ModernBERT-large")
NUM_RUNS=10
EPOCHS=4
BATCH_SIZE=8
DATA_FILE="CR_ECSS_dataset.json"
OUTPUT_DIR="results"
SCRIPT="fine_tune_with_summary.py"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "======================================================================"
echo "STARTING AUTOMATED EXPERIMENTS"
echo "======================================================================"
echo "Configuration:"
echo "  Models: ${MODEL_NAMES[@]}"
echo "  Runs per model: $NUM_RUNS"
echo "  Epochs per run: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Output directory: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Loop through each model
for i in {0..2}; do
    MODEL_PATH=${MODELS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    
    echo ""
    echo "======================================================================"
    echo "MODEL: $MODEL_NAME"
    echo "======================================================================"
    echo "Path: $MODEL_PATH"
    echo "Starting $NUM_RUNS runs..."
    echo ""
    
    # Loop through runs
    for run in $(seq 1 $NUM_RUNS); do
        SEED=$((42 + run - 1))
        
        echo "----------------------------------------------------------------------"
        echo "Run $run/$NUM_RUNS | Seed: $SEED | Model: $MODEL_NAME"
        echo "----------------------------------------------------------------------"
        
        # Run training
        python $SCRIPT \
            --model-path $MODEL_PATH \
            --data-file $DATA_FILE \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --seed $SEED \
            --run-number $run \
            --output-dir $OUTPUT_DIR
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "SUCCESS: Run $run completed for $MODEL_NAME"
        else
            echo "ERROR: Run $run failed for $MODEL_NAME (exit code: $EXIT_CODE)"
            echo "Continuing with next run..."
        fi
        
        echo ""
    done
    
    echo "======================================================================"
    echo "COMPLETED ALL RUNS FOR: $MODEL_NAME"
    echo "Summary file: $OUTPUT_DIR/${MODEL_NAME}_run_summary.csv"
    echo "======================================================================"
    echo ""
done

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo ""
echo "Summary files created:"
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    SUMMARY_FILE="$OUTPUT_DIR/${MODEL_NAME}_run_summary.csv"
    if [ -f "$SUMMARY_FILE" ]; then
        NUM_ROWS=$(wc -l < "$SUMMARY_FILE")
        echo "  $SUMMARY_FILE ($((NUM_ROWS - 1)) runs)"
    else
        echo "  $SUMMARY_FILE (NOT FOUND)"
    fi
done

echo ""
echo "Next steps:"
echo "  1. Analyze results: python analyze_results.py"
echo "  2. View summaries: cat $OUTPUT_DIR/*_run_summary.csv"
echo ""
echo "======================================================================"
