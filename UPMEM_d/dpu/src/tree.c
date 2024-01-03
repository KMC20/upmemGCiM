/*
Author: KMC20
Date: 2023/12/1
Function: Operations for the tree building phase of GCiM.
*/

#include "tree.h"

BARRIER_INIT(barrier_tree, NR_TASKLETS);
MUTEX_INIT(mutex_sums);

// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
static uint64_t maskBase = (1 << (sizeof(ELEMTYPE) << 3)) - 1;
static uint32_t rightShiftBase = sizeof(uint64_t) / sizeof(ELEMTYPE);  // Assume that sizeof(ELEMTYPE) is always no larger than sizeof(uint64_t) here!
// #endif

MEAN_VALUE_TYPE accumulator(const ADDRTYPE left, const ADDRTYPE right, const uint32_t dim, const uint32_t dimAmt) {  // Use DPU_MRAM_HEAP_POINTER to point to points; Reduce multi-thread results on top
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
    uint64_t mask = maskBase << (dim % rightShiftBase << 3);  // Assume that the size of each point is always a multiple of 8, and the start address of points is always aliged on 8 bytes! This is alright for SIFT/GIST/DEEP datasets used for tests
    uint32_t rightShift = dim % rightShiftBase << 3;
    uint64_t maskAddr = ~(uint64_t)(MRAM_ALIGN_BYTES - 1);
// #endif
    uint32_t stride = NR_TASKLETS * dimAmt;
    ADDRTYPE multiLeft = left + me();
    MEAN_VALUE_TYPE sum = 0;
    for (__mram_ptr ELEMTYPE *pointPt = (__mram_ptr ELEMTYPE *)DPU_MRAM_HEAP_POINTER + dimAmt * multiLeft + dim, *pointPtEnd = (__mram_ptr ELEMTYPE *)DPU_MRAM_HEAP_POINTER + dimAmt * right; pointPt < pointPtEnd; pointPt += stride) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
        __dma_aligned uint64_t elem;
        mram_read((__mram_ptr ELEMTYPE *)((uint64_t)pointPt & maskAddr), &elem, sizeof(elem));  // Aligned read: multi-thread safe
        sum += (elem & mask) >> rightShift;  // Little-endian
// #else
//         __dma_aligned ELEMTYPE elem;
//         mram_read(pointPt, &elem, sizeof(elem));
//         sum += elem;
// #endif
    }
    return sum;
}

ADDRTYPE meanSpliter(const ADDRTYPE left, const ADDRTYPE right, const MEAN_VALUE_TYPE mean, const uint32_t dim, const uint32_t dimAmt) {  // Return the index of the last element that is smaller than or equal to the mean value
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
    uint64_t mask = maskBase << (dim % rightShiftBase << 3);  // Assume that the size of each point is always a multiple of 8, and the start address of points is always aliged on 8 bytes! This is alright for SIFT/GIST/DEEP datasets used for tests
    uint32_t rightShift = dim % rightShiftBase << 3;
    uint64_t maskAddr = ~(uint64_t)(MRAM_ALIGN_BYTES - 1);
    __dma_aligned uint64_t readbuf;
    ELEMTYPE lbuf, rbuf;
// #else
//     __dma_aligned ELEMTYPE lbuf, rbuf;
// #endif
    ADDRTYPE pivot = left;
    __mram_ptr ELEMTYPE *lBorder = (__mram_ptr ELEMTYPE *)DPU_MRAM_HEAP_POINTER + dimAmt * left + dim, *rBorder = (__mram_ptr ELEMTYPE *)DPU_MRAM_HEAP_POINTER + dimAmt * right;  // Add `dim` to `lBorder` so that the case that `rPt` gets smaller than points[left] would never occur so that no underflow happens
    __mram_ptr ELEMTYPE *lPt = lBorder, *rPt = rBorder - dimAmt + dim;
    uint32_t pointSize = sizeof(ELEMTYPE) * dimAmt;  // Assume that the size of each point (i.e. pointSize) is always a multiple of 8, and the start address of points is always aliged on 8 bytes!
    fsb_allocator_t tmplAllocator = fsb_alloc(pointSize, 1);
    __dma_aligned ELEMTYPE *tmpl = fsb_get(tmplAllocator);
    fsb_allocator_t tmprAllocator = fsb_alloc(pointSize, 1);
    __dma_aligned ELEMTYPE *tmpr = fsb_get(tmprAllocator);
    uint32_t meanEqCnt = 0;
    while (true) {
        while (lPt < rBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
            mram_read((__mram_ptr ELEMTYPE *)((uint64_t)lPt & maskAddr), &readbuf, sizeof(readbuf));
            lbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(lPt, &lbuf, sizeof(lbuf));
// #endif
            if (lbuf <= mean)
                lPt += dimAmt, ++pivot, meanEqCnt += lbuf == mean;
            else
                break;
        }
        while (rPt > lBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
            mram_read((__mram_ptr ELEMTYPE *)((uint64_t)rPt & maskAddr), &readbuf, sizeof(readbuf));
            rbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(rPt, &rbuf, sizeof(rbuf));
// #endif
            if (rbuf > mean)
                rPt -= dimAmt;
            else
                break;
        }
        if (lPt < rPt) {
            mram_read(lPt - dim, tmpl, pointSize);
            mram_read(rPt - dim, tmpr, pointSize);
            mram_write(tmpl, rPt - dim, pointSize);
            mram_write(tmpr, lPt - dim, pointSize);
            lPt += dimAmt, ++pivot, meanEqCnt += *(tmpr + dim) == mean, rPt -= dimAmt;
        } else {
            break;
        }
    }
    if (meanEqCnt * dimAmt > (uint32_t)(rBorder - rPt))  {  // Solve the extreme imbalance problem
        meanEqCnt >>= 1;  // The amount of points that will be added into the right part
        lPt = lBorder;
        if (rPt <= lBorder)
            rPt += dimAmt;
        while (rPt < rBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
            mram_read((__mram_ptr ELEMTYPE *)((uint64_t)rPt & maskAddr), &readbuf, sizeof(readbuf));
            rbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(rPt, &rbuf, sizeof(rbuf));
// #endif
            if (rbuf > mean)
                break;
            else
                rPt += dimAmt;
        }
        if (rPt > rBorder - dimAmt + dim)
            rPt = rBorder - dimAmt + dim;
        pivot -= meanEqCnt;
        while (meanEqCnt > 0) {
            while (lPt < rBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
                mram_read((__mram_ptr ELEMTYPE *)((uint64_t)lPt & maskAddr), &readbuf, sizeof(readbuf));
                lbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(lPt, &lbuf, sizeof(lbuf));
// #endif
                if (lbuf < mean)
                    lPt += dimAmt;
                else
                    break;  // Expect that lbuf == mean
            }
            while (rPt > lBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
                mram_read((__mram_ptr ELEMTYPE *)((uint64_t)rPt & maskAddr), &readbuf, sizeof(readbuf));
                rbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(rPt, &rbuf, sizeof(rbuf));
// #endif
                if (rbuf > mean)
                    rPt -= dimAmt;
                else
                    break;  // Expect that rbuf <= mean
            }
            if (lPt < rPt) {
                mram_read(lPt - dim, tmpl, pointSize);
                mram_read(rPt - dim, tmpr, pointSize);
                mram_write(tmpl, rPt - dim, pointSize);
                mram_write(tmpr, lPt - dim, pointSize);
                rPt -= dimAmt;
                --meanEqCnt;
            } else {
                break;
            }
        }
    }
    fsb_free(tmprAllocator, tmpr);
    fsb_free(tmplAllocator, tmpl);
    return pivot;
}

MEAN_VALUE_TYPE accumulatorIndependent(const __mram_ptr ELEMTYPE *const points, const ADDRTYPE left, const ADDRTYPE right, const uint32_t dim, const uint32_t dimAmt) {  // Reduce multi-thread results on top
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
    uint64_t mask = maskBase << (dim % rightShiftBase << 3);  // Assume that the size of each point is always a multiple of 8, and the start address of points is always aliged on 8 bytes! This is alright for SIFT/GIST/DEEP datasets used for tests
    uint32_t rightShift = dim % rightShiftBase << 3;
    uint64_t maskAddr = ~(uint64_t)(MRAM_ALIGN_BYTES - 1);
// #endif
    uint32_t stride = NR_TASKLETS * dimAmt;
    ADDRTYPE multiLeft = left + me();
    MEAN_VALUE_TYPE sum = 0;
    for (__mram_ptr ELEMTYPE *pointPt = (__mram_ptr ELEMTYPE *)points + dimAmt * multiLeft + dim, *pointPtEnd = (__mram_ptr ELEMTYPE *)points + dimAmt * right; pointPt < pointPtEnd; pointPt += stride) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
        __dma_aligned uint64_t elem;
        mram_read((__mram_ptr ELEMTYPE *)((uint64_t)pointPt & maskAddr), &elem, sizeof(elem));  // Aligned read: multi-thread safe
        sum += (elem & mask) >> rightShift;  // Little-endian
// #else
//         __dma_aligned ELEMTYPE elem;
//         mram_read(pointPt, &elem, sizeof(elem));
//         sum += elem;
// #endif
    }
    return sum;
}

ADDRTYPE meanSpliterIndependent(__mram_ptr ELEMTYPE *points, __dma_aligned ELEMTYPE *const tmpl, __dma_aligned ELEMTYPE *const tmpr, const uint32_t pointSize, const ADDRTYPE left, const ADDRTYPE right, const MEAN_VALUE_TYPE mean, const uint32_t dim, const uint32_t dimAmt) {  // Return the index of the last element that is smaller than or equal to the mean value
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
    uint64_t mask = maskBase << (dim % rightShiftBase << 3);  // Assume that the size of each point is always a multiple of 8, and the start address of points is always aliged on 8 bytes! This is alright for SIFT/GIST/DEEP datasets used for tests
    uint32_t rightShift = dim % rightShiftBase << 3;
    uint64_t maskAddr = ~(uint64_t)(MRAM_ALIGN_BYTES - 1);
    __dma_aligned uint64_t readbuf;
    ELEMTYPE lbuf, rbuf;
// #else
//     __dma_aligned ELEMTYPE lbuf, rbuf;
// #endif
    ADDRTYPE pivot = left;
    __mram_ptr ELEMTYPE *lBorder = points + dimAmt * left + dim, *rBorder = points + dimAmt * right;  // Add `dim` to `lBorder` so that the case that `rPt` gets smaller than points[left] would never occur so that no underflow happens
    __mram_ptr ELEMTYPE *lPt = lBorder, *rPt = rBorder - dimAmt + dim;
    uint32_t meanEqCnt = 0;
    while (true) {
        while (lPt < rBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
            mram_read((__mram_ptr ELEMTYPE *)((uint64_t)lPt & maskAddr), &readbuf, sizeof(readbuf));
            lbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(lPt, &lbuf, sizeof(lbuf));
// #endif
            if (lbuf <= mean)
                lPt += dimAmt, ++pivot, meanEqCnt += lbuf == mean;
            else
                break;
        }
        while (rPt > lBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
            mram_read((__mram_ptr ELEMTYPE *)((uint64_t)rPt & maskAddr), &readbuf, sizeof(readbuf));
            rbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(rPt, &rbuf, sizeof(rbuf));
// #endif
            if (rbuf > mean)
                rPt -= dimAmt;
            else
                break;
        }
        if (lPt < rPt) {
            mram_read(lPt - dim, tmpl, pointSize);
            mram_read(rPt - dim, tmpr, pointSize);
            mram_write(tmpl, rPt - dim, pointSize);
            mram_write(tmpr, lPt - dim, pointSize);
            lPt += dimAmt, ++pivot, meanEqCnt += *(tmpr + dim) == mean, rPt -= dimAmt;
        } else {
            break;
        }
    }
    if (meanEqCnt * dimAmt > (uint32_t)(rBorder - rPt))  {  // Solve the extreme imbalance problem
        meanEqCnt >>= 1;  // The amount of points that will be added into the right part
        lPt = lBorder;
        if (rPt <= lBorder)
            rPt += dimAmt;
        while (rPt < rBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
            mram_read((__mram_ptr ELEMTYPE *)((uint64_t)rPt & maskAddr), &readbuf, sizeof(readbuf));
            rbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(rPt, &rbuf, sizeof(rbuf));
// #endif
            if (rbuf > mean)
                break;
            else
                rPt += dimAmt;
        }
        if (rPt > rBorder - dimAmt + dim)
            rPt = rBorder - dimAmt + dim;
        pivot -= meanEqCnt;
        while (meanEqCnt > 0) {
            while (lPt < rBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
                mram_read((__mram_ptr ELEMTYPE *)((uint64_t)lPt & maskAddr), &readbuf, sizeof(readbuf));
                lbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(lPt, &lbuf, sizeof(lbuf));
// #endif
                if (lbuf < mean)
                    lPt += dimAmt;
                else
                    break;  // Expect that lbuf == mean
            }
            while (rPt > lBorder) {
// #if (sizeof(ELEMTYPE) < sizeof(uint64_t))
                mram_read((__mram_ptr ELEMTYPE *)((uint64_t)rPt & maskAddr), &readbuf, sizeof(readbuf));
                rbuf = (readbuf & mask) >> rightShift;  // Little-endian
// #else
//             mram_read(rPt, &rbuf, sizeof(rbuf));
// #endif
                if (rbuf > mean)
                    rPt -= dimAmt;
                else
                    break;  // Expect that rbuf <= mean
            }
            if (lPt < rPt) {
                mram_read(lPt - dim, tmpl, pointSize);
                mram_read(rPt - dim, tmpr, pointSize);
                mram_write(tmpl, rPt - dim, pointSize);
                mram_write(tmpr, lPt - dim, pointSize);
                rPt -= dimAmt;
                --meanEqCnt;
            } else {
                break;
            }
        }
    }
    return pivot;
}

#define LCG_M 49381  // Multiplier
#define LCG_I 8643   // Increment
inline unsigned short randGen(const unsigned short dimAmt, uint32_t *randSeed) {  // Since `srand` and `rand` are not implemented by DPU, use pseudo random generator here. Refer to the prand in rand.c on https://github.com/0/msp430-rng. Thanks!
    *randSeed = LCG_M * *randSeed + LCG_I;
    return *randSeed % dimAmt;
}

// Shared varibles in `treeConstrDPU`
ADDRTYPE treeConstrDPU_ltop;
ADDRTYPE treeConstrDPU_rtop;
unsigned short treeConstrDPU_dim;
uint32_t treeConstrDPU_stackSize;
MEAN_VALUE_TYPE treeConstrDPU_sum;
uint32_t treeConstrDPU_meet_leaf;
void treeConstrDPU(treeNode_t *tree, ADDRTYPE *treeSizeRes, __mram_ptr ELEMTYPE *points, const ADDRTYPE treeBaseAddr, const ADDRTYPE pointAmt, const unsigned short dimAmt, const unsigned short leafCapacity) {  // Static linked-list
    if (pointAmt < 1) {
        *treeSizeRes = 0;
        return;
    }
    uint32_t STACK_MAX_SIZE = ((sizeof(uint32_t) << 3) - __builtin_clz(pointAmt)) << 1;  // ceil(log2(pointAmt)) * 2
    treeNode_t *localTree = me() == 0 ? tree - treeBaseAddr : NULL;
    ADDRTYPE treeSize = treeBaseAddr;
    fsb_allocator_t tstackAllocator = me() == 0 ? fsb_alloc(sizeof(ADDRTYPE) * STACK_MAX_SIZE, 1) : NULL;
    __dma_aligned ELEMTYPE *tstack = me() == 0 ? fsb_get(tstackAllocator) : NULL;
    fsb_allocator_t lstackAllocator = me() == 0 ? fsb_alloc(sizeof(ADDRTYPE) * STACK_MAX_SIZE, 1) : NULL;
    __dma_aligned ELEMTYPE *lstack = me() == 0 ? fsb_get(lstackAllocator) : NULL;
    fsb_allocator_t rstackAllocator = me() == 0 ? fsb_alloc(sizeof(ADDRTYPE) * STACK_MAX_SIZE, 1) : NULL;
    __dma_aligned ELEMTYPE *rstack = me() == 0 ? fsb_get(rstackAllocator) : NULL;
    if (me() == 0) {
        treeConstrDPU_stackSize = 0;
        tstack[treeConstrDPU_stackSize] = treeSize++;
        lstack[treeConstrDPU_stackSize] = 0;
        rstack[treeConstrDPU_stackSize++] = pointAmt;
    }
    uint32_t randSeed = (uint32_t)&treeConstrDPU_stackSize + pointAmt;  // Expect the address of `treeConstrDPU_stackSize` is random at each launching. Besides, since the top tree may be built at the host side, the `pointAmt` may be different due to the imbalance split by means. Expect the seed can be different at each launching.
    uint32_t pointSize = sizeof(ELEMTYPE) * dimAmt;
    fsb_allocator_t tmplAllocator = me() == 0 ? fsb_alloc(pointSize, 1) : NULL;
    __dma_aligned ELEMTYPE *tmpl = me() == 0 ? fsb_get(tmplAllocator) : NULL;
    fsb_allocator_t tmprAllocator = me() == 0 ? fsb_alloc(pointSize, 1) : NULL;
    __dma_aligned ELEMTYPE *tmpr = me() == 0 ? fsb_get(tmprAllocator) : NULL;
    barrier_wait(&barrier_tree);
    while (treeConstrDPU_stackSize > 0) {  // Inorder tranverse
        if (me() == 0) {
            --treeConstrDPU_stackSize;
        }
        treeNode_t *ttop = me() == 0 ? localTree + tstack[treeConstrDPU_stackSize] : NULL;
        if (me() == 0) {
            treeConstrDPU_meet_leaf = 0;
            treeConstrDPU_ltop = lstack[treeConstrDPU_stackSize], treeConstrDPU_rtop = rstack[treeConstrDPU_stackSize];
            if (treeConstrDPU_rtop - treeConstrDPU_ltop <= leafCapacity) {  // Leaf node
                ttop->mean = treeConstrDPU_ltop;
                ttop->dim = treeConstrDPU_rtop - treeConstrDPU_ltop;
                ttop->left = ttop->right = ADDRTYPE_NULL;
                treeConstrDPU_meet_leaf = 1;
            }
            treeConstrDPU_dim = randGen(dimAmt, &randSeed);  // The cost of this operation is expected to be less than synchronization, so use barrier after this line
            treeConstrDPU_sum = 0;
        }
        barrier_wait(&barrier_tree);
        if (treeConstrDPU_meet_leaf > 0)
            continue;
        MEAN_VALUE_TYPE sum = accumulatorIndependent(points, treeConstrDPU_ltop, treeConstrDPU_rtop, treeConstrDPU_dim, dimAmt);
        mutex_lock(mutex_sums);
        treeConstrDPU_sum += sum;
        mutex_unlock(mutex_sums);
        barrier_wait(&barrier_tree);
        if (me() == 0) {
            MEAN_VALUE_TYPE mean = treeConstrDPU_sum / (treeConstrDPU_rtop - treeConstrDPU_ltop);
            ADDRTYPE pivot = meanSpliterIndependent(points, tmpl, tmpr, pointSize, treeConstrDPU_ltop, treeConstrDPU_rtop, mean, treeConstrDPU_dim, dimAmt);
            ttop->mean = mean;
            ttop->dim = treeConstrDPU_dim;
            if (treeConstrDPU_rtop - pivot > leafCapacity) {
                ttop->right = treeSize++;
                tstack[treeConstrDPU_stackSize] = ttop->right;
                lstack[treeConstrDPU_stackSize] = pivot;
                rstack[treeConstrDPU_stackSize++] = treeConstrDPU_rtop;
            } else {
                if (pivot < treeConstrDPU_rtop) {
                    ttop->right = treeSize++;
                    treeNode_t *newLeaf = localTree + ttop->right;
                    newLeaf->mean = pivot;
                    newLeaf->dim = treeConstrDPU_rtop - pivot;
                    newLeaf->left = newLeaf->right = ADDRTYPE_NULL;
                } else {
                    ttop->right = ADDRTYPE_NULL;
                }
            }
            if (pivot - treeConstrDPU_ltop > leafCapacity) {
                ttop->left = treeSize++;
                tstack[treeConstrDPU_stackSize] = ttop->left;
                lstack[treeConstrDPU_stackSize] = treeConstrDPU_ltop;
                rstack[treeConstrDPU_stackSize++] = pivot;
            } else {
                if (pivot > treeConstrDPU_ltop) {
                    ttop->left = treeSize++;
                    treeNode_t *newLeaf = localTree + ttop->left;
                    newLeaf->mean = treeConstrDPU_ltop;
                    newLeaf->dim = pivot - treeConstrDPU_ltop;
                    newLeaf->left = newLeaf->right = ADDRTYPE_NULL;
                } else {
                    ttop->left = ADDRTYPE_NULL;
                }
            }
        }
        barrier_wait(&barrier_tree);
    }
    if (me() == 0) {
        fsb_free(tmprAllocator, tmpr);
        fsb_free(tmplAllocator, tmpl);
        fsb_free(rstackAllocator, rstack);
        fsb_free(lstackAllocator, lstack);
        fsb_free(tstackAllocator, tstack);
        *treeSizeRes = treeSize;
    }
}
