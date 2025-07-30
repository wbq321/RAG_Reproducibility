#!/bin/bash
# fixed_quick_test.slurm - Fixed version with proper MPI configuration

#SBATCH --job-name=rag_quick_fixed
#SBATCH --output=quick_fixed_%j.out
#SBATCH --error=quick_fixed_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
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

# Fix 1: Set proper MPI environment variables
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl=^openib

# Fix 2: Check if we need to use different MPI launch method
echo "=== MPI Configuration ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODELIST: $SLURM_NODELIST"

# Test which MPI launcher works
echo -e "\n=== Testing MPI launch methods ==="

# Method 1: Try mpirun instead of srun
echo "Testing mpirun..."
mpirun -n 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}')"

if [ $? -eq 0 ]; then
    echo "mpirun works! Using mpirun for the test."
    MPI_LAUNCHER="mpirun -n $SLURM_NTASKS"
else
    echo "mpirun failed, trying srun with PMI2..."
    # Method 2: Try srun with PMI2
    export SLURM_MPI_TYPE=pmi2
    srun --mpi=pmi2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}')"
    
    if [ $? -eq 0 ]; then
        echo "srun with PMI2 works!"
        MPI_LAUNCHER="srun --mpi=pmi2"
    else
        echo "PMI2 failed, trying PMIx..."
        # Method 3: Try srun with PMIx
        export SLURM_MPI_TYPE=pmix
        srun --mpi=pmix python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}')"
        
        if [ $? -eq 0 ]; then
            echo "srun with PMIx works!"
            MPI_LAUNCHER="srun --mpi=pmix"
        else
            echo "All MPI methods failed. Falling back to single-node test."
            MPI_LAUNCHER="python"
        fi
    fi
fi

# Create output directory
OUTPUT_DIR="quick_results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Save the optimized test script
cat > $OUTPUT_DIR/simple_mpi_test.py << 'EOF'
#!/usr/bin/env python
"""Simple MPI test without complex dependencies"""

import os
import sys
import time
import json
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpi_available = True
except ImportError:
    print("MPI not available, running in single-process mode")
    rank = 0
    size = 1
    mpi_available = False
    comm = None

print(f"Process {rank} of {size} starting...")

# Simple reproducibility test
def simple_faiss_test():
    """Test FAISS without the full framework"""
    try:
        import faiss
        
        # Create simple data
        np.random.seed(42 + rank)
        n_docs = 1000 // size  # Each rank handles subset
        dim = 128  # Smaller dimension for speed
        
        # Generate random embeddings
        embeddings = np.random.randn(n_docs, dim).astype('float32')
        
        # Create simple index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        # Search
        n_queries = 5
        queries = np.random.randn(n_queries, dim).astype('float32')
        k = 10
        
        distances, indices = index.search(queries, k)
        
        print(f"[Rank {rank}] Indexed {n_docs} docs, searched {n_queries} queries")
        
        # Simple reproducibility check
        distances2, indices2 = index.search(queries, k)
        
        reproducible = np.array_equal(indices, indices2)
        print(f"[Rank {rank}] Reproducible: {reproducible}")
        
        return {
            "rank": rank,
            "n_docs": n_docs,
            "reproducible": reproducible,
            "first_result": indices[0].tolist() if len(indices) > 0 else []
        }
        
    except ImportError:
        print(f"[Rank {rank}] FAISS not available")
        return {"rank": rank, "error": "FAISS not available"}
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        return {"rank": rank, "error": str(e)}

# Run test
start_time = time.time()
result = simple_faiss_test()
elapsed = time.time() - start_time

result["elapsed_time"] = elapsed

# Gather results if MPI available
if mpi_available and comm is not None:
    all_results = comm.gather(result, root=0)
    
    if rank == 0:
        print("\n=== All Results ===")
        for r in all_results:
            print(f"Rank {r['rank']}: {r}")
        
        # Save results
        with open("test_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nTotal test time: {elapsed:.2f} seconds")
else:
    # Single process mode
    print(f"\nSingle process result: {result}")
    with open("test_results.json", "w") as f:
        json.dump([result], f, indent=2)

print(f"Process {rank} completed")
EOF

# Run the test
echo -e "\n=== Running simple MPI test ==="
cd $OUTPUT_DIR
$MPI_LAUNCHER python simple_mpi_test.py

# Check results
echo -e "\n=== Test Results ==="
if [ -f "test_results.json" ]; then
    cat test_results.json
else
    echo "No results file found"
fi

echo -e "\nJob completed at: $(date)"

# Create a standalone test script for debugging
cat > $OUTPUT_DIR/debug_mpi.sh << 'EOF'
#!/bin/bash
# Standalone MPI debugging script

echo "Testing different MPI configurations..."

# Test 1: Basic Python
echo -e "\n1. Basic Python test:"
python -c "print('Python works')"

# Test 2: MPI4PY import
echo -e "\n2. MPI4PY import test:"
python -c "import mpi4py; print(f'MPI4PY version: {mpi4py.__version__}')"

# Test 3: Different MPI launchers
echo -e "\n3. Testing MPI launchers:"

echo "  a) mpirun:"
mpirun -n 2 hostname

echo "  b) mpiexec:"
mpiexec -n 2 hostname

echo "  c) srun (if in SLURM):"
if [ -n "$SLURM_JOB_ID" ]; then
    srun -n 2 hostname
else
    echo "     Not in SLURM job"
fi

# Test 4: Environment variables
echo -e "\n4. MPI-related environment variables:"
env | grep -E "(MPI|PMI|SLURM)" | sort
EOF

chmod +x $OUTPUT_DIR/debug_mpi.sh

echo -e "\nDebug script created: $OUTPUT_DIR/debug_mpi.sh"
echo "You can run it manually to diagnose MPI issues"