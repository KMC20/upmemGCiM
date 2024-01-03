/*
Author: KMC20
Date: 2023/12/6
Function: Entry to the component accelerator of the tree building phase of GCiM on DPUs.
*/

#include "tree.h"

BARRIER_INIT(barrier_TBP_accumulator, NR_TASKLETS);
MUTEX_INIT(mutex_sumRes);

#define POINT_MEM_SIZE (62 << 20)

// Inputs
__mram_noinit ELEMTYPE points[POINT_MEM_SIZE / sizeof(ELEMTYPE)];  // Annotate this line if using DPU_MRAM_HEAP_POINTER to point to points
__host ADDRTYPE pointAmt;
__host uint32_t dim;
__host uint32_t dimAmt;
// Outputs
__host MEAN_VALUE_TYPE sumRes;
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

    if (me() == 0) {
        sumRes = 0;
    }
    barrier_wait(&barrier_TBP_accumulator);
    if (pointAmt < 1)
        return 0;
    MEAN_VALUE_TYPE sumResMe = accumulatorIndependent(points, 0, pointAmt, dim, dimAmt);
    mutex_lock(mutex_sumRes);
    sumRes += sumResMe;
    mutex_unlock(mutex_sumRes);
    
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
