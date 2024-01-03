/*
Author: KMC20
Date: 2023/12/6
Function: Entry to the component mean splitter of the tree building phase of GCiM on DPUs.
*/

#include "tree.h"

#define POINT_MEM_SIZE (62 << 20)

// Inputs
__mram_noinit ELEMTYPE points[POINT_MEM_SIZE / sizeof(ELEMTYPE)];  // Annotate this line if using DPU_MRAM_HEAP_POINTER to point to points
__host ADDRTYPE pointAmt;
__host MEAN_VALUE_TYPE mean;
__host uint32_t dim;
__host uint32_t dimAmt;
// Outputs
__host ADDRTYPE splitRes;
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

    if (pointAmt < 1) {
        splitRes = 0;
        return 0;
    }

    if (me() == 0) {
        uint32_t pointSize = sizeof(ELEMTYPE) * dimAmt;
        fsb_allocator_t tmplAllocator = fsb_alloc(pointSize, 1);
        __dma_aligned ELEMTYPE *tmpl = fsb_get(tmplAllocator);
        fsb_allocator_t tmprAllocator = fsb_alloc(pointSize, 1);
        __dma_aligned ELEMTYPE *tmpr = fsb_get(tmprAllocator);
        splitRes = meanSpliterIndependent(points, tmpl, tmpr, pointSize, 0, pointAmt, mean, dim, dimAmt);
        fsb_free(tmprAllocator, tmpr);
        fsb_free(tmplAllocator, tmpl);
    }

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
