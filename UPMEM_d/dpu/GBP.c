/*
Author: KMC20
Date: 2023/12/6
Function: Entry to the graph building phase of GCiM on DPUs.
*/

#include "graph.h"
#include <stdio.h>

#define POINT_MEM_SIZE (60 << 20)
#define NEIGHBOR_MEM_SIZE (2 << 20)

// Inputs
__host uint32_t pointAmt;
__host uint32_t dimAmt;
__mram_noinit ELEMTYPE points[POINT_MEM_SIZE / sizeof(ELEMTYPE)];
__host uint32_t neighborAmt;
// Outputs
__mram_noinit pqueue_elem_t_mram neighbors[NEIGHBOR_MEM_SIZE / sizeof(pqueue_elem_t_mram)];
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
    graphBuilding(points, pointAmt, dimAmt, neighborAmt, 0, neighbors);
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
