/*
Author: KMC20
Date: 2023/12/6
Function: Entry to the tree building phase of GCiM on DPUs.
*/

#include "tree.h"

#define POINT_MEM_SIZE (62 << 20)
#define TREE_MEM_SIZE  (50 << 10)

// Inputs
__host ADDRTYPE pointAmt;
__host uint32_t dimAmt;
__host uint32_t leafCapacity;
// Inouts
__mram_noinit ELEMTYPE points[POINT_MEM_SIZE / sizeof(ELEMTYPE)];  // Annotate this line if using DPU_MRAM_HEAP_POINTER to point to points
// Outputs
__host treeNode_t tree[TREE_MEM_SIZE / sizeof(treeNode_t)];
__host ADDRTYPE treeSizeRes;
#ifdef PERF_EVAL_SIM
__host perfcounter_t exec_time;
MUTEX_INIT(mutex_exec_time);
#endif

int main() {
#ifdef PERF_EVAL_SIM
    if (me() == 0) {
        exec_time = 0;
        perfcounter_config(COUNT_CYCLES, true);  // `The main difference between counting cycles and instructions is that cycles include the execution time of instructions AND the memory transfers.`
        // perfcounter_config(COUNT_INSTRUCTIONS, true);
    }
#endif
    treeConstrDPU(tree, &treeSizeRes, points, 0, pointAmt, dimAmt, leafCapacity);
#ifdef PERF_EVAL_SIM
    perfcounter_t exec_time_me = perfcounter_get();
    mutex_lock(mutex_exec_time);
    if (exec_time_me > exec_time) {
        exec_time = exec_time_me;
    }
    mutex_unlock(mutex_exec_time);
#endif
    return 0;
}

