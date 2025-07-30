#!/bin/bash
# quick_test.slurm - Quick test to validate setup

#SBATCH --job-name=rag_quick
#SBATCH --output=quick_%j.out
#SBATCH --error=quick_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load modules
module load python/3.9
module load cuda/11.8
module load openmpi/4.1.4

# Activate environment
source ~/rag_env/bin/activate

# Set environment for deterministic behavior
export CUDA_LAUNCH_BLOCKING=1
export PYTHONHASHSEED=0

# Create output directory
OUTPUT_DIR="quick_results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Log system info
echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"

# First, run the debug test
echo -e "\n=== Running debug test ==="
sbatch debug_test.slurm

# Run the optimized quick test
echo -e "\n=== Running optimized quick test ==="
srun python optimized_small_test.py $OUTPUT_DIR

echo -e "\nJob completed at: $(date)"
echo "Results saved in: $OUTPUT_DIR"

# Print summary
if [ -f "$OUTPUT_DIR/quick_test_results.json" ]; then
    echo -e "\n=== Test Summary ==="
    python -c "
import json
with open('$OUTPUT_DIR/quick_test_results.json', 'r') as f:
    data = json.load(f)
    metrics = data['metrics']
    print(f'Exact Match: {metrics[\"exact_match\"][\"exact_match_rate\"]:.3f}')
    print(f'Jaccard: {metrics[\"overlap\"][\"mean_jaccard\"]:.3f}')
"
fi