/*
Author: KMC20
Date: 2023/12/16
Function: Operations for the tree building phase on host of GCiM.
*/

#include "tree.h"

MEAN_VALUE_TYPE accumulatorIndependent(const ELEMTYPE *const points, const ADDRTYPE left, const ADDRTYPE right, const uint32_t dim, const uint32_t dimAmt) {
    MEAN_VALUE_TYPE sum = 0;
    for (ELEMTYPE *pointPt = (ELEMTYPE *)points + dimAmt * left + dim, *pointPtEnd = (ELEMTYPE *)points + dimAmt * right; pointPt < pointPtEnd; pointPt += dimAmt) {
        sum += *pointPt;
    }
    return sum;
}

ADDRTYPE meanSpliterIndependent(ELEMTYPE *points, const uint32_t pointSize, const ADDRTYPE left, const ADDRTYPE right, const MEAN_VALUE_TYPE mean, const uint32_t dim, const uint32_t dimAmt) {  // Return the index of the last element that is smaller than or equal to the mean value
    ADDRTYPE pivot = left;
    ELEMTYPE *lBorder = points + dimAmt * left + dim, *rBorder = points + dimAmt * right;  // Add `dim` to `lBorder` so that the case that `rPt` gets smaller than points[left] would never occur so that no underflow happens
    ELEMTYPE *lPt = lBorder, *rPt = rBorder - dimAmt + dim;
    uint32_t meanEqCnt = 0;
    while (true) {
        while (lPt < rBorder) {
            if (*lPt <= mean)
                lPt += dimAmt, ++pivot, meanEqCnt += *lPt == mean;
            else
                break;
        }
        while (rPt > lBorder) {
            if (*rPt > mean)
                rPt -= dimAmt;
            else
                break;
        }
        if (lPt < rPt) {
            ELEMTYPE tmp[dimAmt];
            ELEMTYPE *lPointPt = lPt - dim, *rPointPt = rPt - dim;
            memcpy(tmp, lPointPt, pointSize);
            memcpy(lPointPt, rPointPt, pointSize);
            memcpy(rPointPt, tmp, pointSize);
            lPt += dimAmt, ++pivot, meanEqCnt += *lPt == mean, rPt -= dimAmt;
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
            if (*rPt > mean)
                break;
            else
                rPt += dimAmt;
        }
        if (rPt > rBorder - dimAmt + dim)
            rPt = rBorder - dimAmt + dim;
        pivot -= meanEqCnt;
        while (meanEqCnt > 0) {
            while (lPt < rBorder) {
                if (*lPt < mean)
                    lPt += dimAmt;
                else
                    break;  // Expect that *lPt == mean
            }
            while (rPt > lBorder) {
                if (*rPt > mean)
                    rPt -= dimAmt;
                else
                    break;  // Expect that *rPt <= mean
            }
            if (lPt < rPt) {
                ELEMTYPE tmp[dimAmt];
                ELEMTYPE *lPointPt = lPt - dim, *rPointPt = rPt - dim;
                memcpy(tmp, lPointPt, pointSize);
                memcpy(lPointPt, rPointPt, pointSize);
                memcpy(rPointPt, tmp, pointSize);
                rPt -= dimAmt;
                --meanEqCnt;
            } else {
                break;
            }
        }
    }
    return pivot;
}

void treeConstrDPU(treeNode_t *tree, ADDRTYPE *treeSizeRes, ELEMTYPE *points, const ADDRTYPE treeBaseAddr, const ADDRTYPE pointAmt, const unsigned short dimAmt, const unsigned short leafCapacity, ADDRTYPE *leafIds, ADDRTYPE *leafIdSizeRes) {  // Static linked-list
    if (pointAmt < 1)
        return;
    uint32_t STACK_MAX_SIZE = ((sizeof(uint32_t) << 3) - __builtin_clz(pointAmt)) << 1;  // ceil(log2(pointAmt)) * 2
    treeNode_t *localTree = tree - treeBaseAddr;
    ADDRTYPE treeSize = treeBaseAddr;
    ADDRTYPE leafSize = 0;
    ADDRTYPE tstack[STACK_MAX_SIZE << 1];
    ADDRTYPE lstack[STACK_MAX_SIZE << 1];
    ADDRTYPE rstack[STACK_MAX_SIZE << 1];
    uint32_t stackSize = 0;
    tstack[stackSize] = treeSize++;
    lstack[stackSize] = 0;
    rstack[stackSize++] = pointAmt;
    srand(time(0));
    uint32_t pointSize = sizeof(ELEMTYPE) * dimAmt;
    while (stackSize > 0) {  // Inorder tranverse
        treeNode_t *ttop = localTree + tstack[--stackSize];
        ADDRTYPE ltop = lstack[stackSize], rtop = rstack[stackSize];
        if (rtop - ltop <= leafCapacity) {  // Leaf node
            ttop->mean = ltop;
            ttop->dim = rtop - ltop;
            ttop->left = ttop->right = ADDRTYPE_NULL;
            leafIds[leafSize++] = ttop - localTree;
            continue;
        }
        unsigned short dim = rand() % dimAmt;
        MEAN_VALUE_TYPE sum = accumulatorIndependent(points, ltop, rtop, dim, dimAmt);
        MEAN_VALUE_TYPE mean = sum / (rtop - ltop);
        ADDRTYPE pivot = meanSpliterIndependent(points, pointSize, ltop, rtop, mean, dim, dimAmt);
        ttop->mean = mean;
        ttop->dim = dim;
        if (rtop - pivot > leafCapacity) {
            ttop->right = treeSize++;
            tstack[stackSize] = ttop->right;
            lstack[stackSize] = pivot;
            rstack[stackSize++] = rtop;
        } else {
            if (pivot < rtop) {
                ttop->right = treeSize++;
                treeNode_t *newLeaf = localTree + ttop->right;
                newLeaf->mean = pivot;
                newLeaf->dim = rtop - pivot;
                newLeaf->left = newLeaf->right = ADDRTYPE_NULL;
                leafIds[leafSize++] = ttop->right;
            } else {
                ttop->right = ADDRTYPE_NULL;
            }
        }
        if (pivot - ltop > leafCapacity) {
            ttop->left = treeSize++;
            tstack[stackSize] = ttop->left;
            lstack[stackSize] = ltop;
            rstack[stackSize++] = pivot;
        } else {
            if (pivot > ltop) {
                ttop->left = treeSize++;
                treeNode_t *newLeaf = localTree + ttop->left;
                newLeaf->mean = ltop;
                newLeaf->dim = pivot - ltop;
                newLeaf->left = newLeaf->right = ADDRTYPE_NULL;
                leafIds[leafSize++] = ttop->left;
            } else {
                ttop->left = ADDRTYPE_NULL;
            }
        }
    }
    *treeSizeRes = treeSize, *leafIdSizeRes = leafSize;
}
