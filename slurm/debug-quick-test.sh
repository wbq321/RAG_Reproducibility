#!/bin/bash
# debug_test.slurm - Quick debugging test for cluster

#SBATCH --job-name=rag_debug
#SBATCH --output=debug_%j.out
#SBATCH --error=debug_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load modules
module load python/3.9 cuda/11.8 openmpi/4.1.4

# Activate environment
source ~/rag_env/bin/activate

# Print debug information
echo "========== DEBUG INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs per node: 1"
echo "Start time: $(date)"
echo "==============================="

# Test MPI
echo -e "\n=== Testing MPI ==="
mpirun --version
mpirun -n $SLURM_NTASKS hostname

# Test GPU access
echo -e "\n=== Testing GPU ==="
nvidia-smi

# Test Python imports
echo -e "\n=== Testing Python imports ==="
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError as e:
    print(f'NumPy import failed: {e}')
    
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'PyTorch import failed: {e}')
    
try:
    import faiss
    print(f'FAISS available')
    print(f'FAISS GPU support: {faiss.get_num_gpus()}')
except ImportError as e:
    print(f'FAISS import failed: {e}')
    
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print(f'MPI4PY: rank {comm.Get_rank()} of {comm.Get_size()}')
except ImportError as e:
    print(f'MPI4PY import failed: {e}')
"

# Run minimal test
echo -e "\n=== Running minimal test ==="
OUTPUT_DIR="debug_results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Create minimal test script
cat > $OUTPUT_DIR/minimal_test.py << 'EOF'
import time
import json
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"[Rank {rank}] Starting minimal test...")

# Simple timing test
start = time.time()

# Test data creation
if rank == 0:
    data = {"test": "data", "size": 100}
else:
    data = None

# Broadcast test
data = comm.bcast(data, root=0)
print(f"[Rank {rank}] Received data: {data}")

# Simple computation
result = sum(range(1000000))
print(f"[Rank {rank}] Computation result: {result}")

# Gather results
results = comm.gather(result, root=0)

if rank == 0:
    print(f"All results: {results}")
    
    # Save timing
    elapsed = time.time() - start
    with open("timing.json", "w") as f:
        json.dump({"elapsed": elapsed, "ranks": size}, f)
    
    print(f"Total time: {elapsed:.2f} seconds")

comm.Barrier()
print(f"[Rank {rank}] Done")
EOF

# Run the minimal test
cd $OUTPUT_DIR
time mpirun -n $SLURM_NTASKS python minimal_test.py

echo -e "\n=== Test completed ==="
echo "End time: $(date)"
echo "Results in: $OUTPUT_DIR"