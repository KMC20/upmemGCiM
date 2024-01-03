/*
Author: KMC20
Date: 2023/12/1
Function: Operations for the tree building phase of GCiM.
*/

#ifndef GCIM_TREE_H
#define GCIM_TREE_H

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>
#include <stdint.h>
#include "request.h"

MEAN_VALUE_TYPE accumulator(const ADDRTYPE left, const ADDRTYPE right, const uint32_t dim, const uint32_t dimAmt);
ADDRTYPE meanSpliter(const ADDRTYPE left, const ADDRTYPE right, const MEAN_VALUE_TYPE mean, const uint32_t dim, const uint32_t dimAmt);
MEAN_VALUE_TYPE accumulatorIndependent(const __mram_ptr ELEMTYPE *const points, const ADDRTYPE left, const ADDRTYPE right, const uint32_t dim, const uint32_t dimAmt);
ADDRTYPE meanSpliterIndependent(__mram_ptr ELEMTYPE *points, __dma_aligned ELEMTYPE *const tmpl, __dma_aligned ELEMTYPE *const tmpr, const uint32_t pointSize, const ADDRTYPE left, const ADDRTYPE right, const MEAN_VALUE_TYPE mean, const uint32_t dim, const uint32_t dimAmt);
void treeConstrDPU(treeNode_t *tree, ADDRTYPE *treeSizeRes, __mram_ptr ELEMTYPE *points, const ADDRTYPE treeBaseAddr, const uint32_t pointAmt, const unsigned short dimAmt, const unsigned short leafCapacity);

#endif