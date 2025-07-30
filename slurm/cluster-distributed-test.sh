#!/bin/bash
# submit_job.slurm - SLURM job submission script for distributed RAG testing

#SBATCH --job-name=rag_reproducibility
#SBATCH --output=rag_test_%j.out
#SBATCH --error=rag_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --exclusive

# Load required modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8
module load openmpi/4.1.4
module load gcc/9.3.0

# Activate virtual environment
source ~/rag_env/bin/activate

# Set environment variables for better reproducibility
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: 2"
echo "Start time: $(date)"

# Create output directory for this job
OUTPUT_DIR="results/job_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Copy configuration file
cp config/cluster_config.json $OUTPUT_DIR/

# Run distributed test
srun python distributed_rag_test.py \
    --output-dir $OUTPUT_DIR \
    --config config/cluster_config.json \
    --num-docs 1000000 \
    --num-queries 1000 \
    --num-runs 10

# Generate consolidated report
if [ $SLURM_PROCID -eq 0 ]; then
    python generate_cluster_report.py --job-dir $OUTPUT_DIR
fi

echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"