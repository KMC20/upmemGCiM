/*
Author: KMC20
Date: 2023/12/1
Function: Operations for the graph building phase of GCiM.
*/

#include "graph.h"

/**************************************************************************************************************************************************************************************************************/
/*                                                              Functions for the usage of priority queue (Copied from `sample.c` of `libpqueue`)                                                             */
/**************************************************************************************************************************************************************************************************************/
static int cmp_pri(pqueue_pri_t next, pqueue_pri_t curr) {
	return (next < curr);
}
static pqueue_pri_t get_pri(void *a) {
	return ((pqueue_elem_t *) a)->pri;
}
static void set_pri(void *a, pqueue_pri_t pri) {
	((pqueue_elem_t *) a)->pri = pri;
}
static size_t get_pos(void *a) {
	return ((pqueue_elem_t *) a)->pos;
}
static void set_pos(void *a, size_t pos) {
	((pqueue_elem_t *) a)->pos = pos;
}	
/***************************************************************************************************************************************************************************************************************/


pqueue_pri_t distCalVec(const ELEMTYPE *const vec1, const ELEMTYPE *const vec2, const unsigned short dimAmt) {
    pqueue_pri_t res = 0, diff;
    ELEMTYPE *vec1End = (ELEMTYPE *)vec1 + dimAmt;
    for (ELEMTYPE *vec1pt = (ELEMTYPE *)vec1, *vec2pt = (ELEMTYPE *)vec2; vec1pt < vec1End; ++vec1pt, ++vec2pt) {
        diff = *vec1pt > *vec2pt ? *vec1pt - *vec2pt : *vec2pt - *vec1pt;
        res += diff * diff;
    }
    return res;
}

MUTEX_INIT(mutex_unaligned_neighbors);
void save_pq_into_mram(pqueue_t *pq, const ADDRTYPE pointNeighborStartAddr, __mram_ptr pqueue_elem_t_mram *pointsMram) {
    uint32_t priUnalignedBytes = pointNeighborStartAddr & MRAM_ALIGN_BYTES - 1;
    if (priUnalignedBytes != 0) {  // Unaligned address. This case should be avoided and hardly happens since the auto alignment of struct data type
        pqueue_elem_t *pqTop = NULL;
        fsb_allocator_t wramWriteBufAllocator = fsb_alloc(sizeof(pqueue_elem_t_mram) * pq->size, 1);
        __dma_aligned uint8_t *wramWriteBuf = fsb_get(wramWriteBufAllocator);
        uint32_t wramWriteBufSize = 0;
        for (; wramWriteBufSize < priUnalignedBytes && (pqTop = pqueue_pop(pq)) != NULL; wramWriteBufSize += sizeof(pqueue_elem_t_mram))
            memcpy(wramWriteBuf, pqTop, sizeof(pqueue_elem_t_mram));
        __dma_aligned uint64_t priBuf;
        __mram_ptr pqueue_elem_t_mram *pointsMramAligned = pointsMram + pointNeighborStartAddr - priUnalignedBytes;
        uint32_t priCopiedBytes = MRAM_ALIGN_BYTES - priUnalignedBytes;
        mutex_lock(mutex_unaligned_neighbors);
        mram_read(pointsMramAligned, &priBuf, sizeof(uint64_t));
        memcpy((uint8_t *)&priBuf + priUnalignedBytes, wramWriteBuf, priCopiedBytes);
        mram_write(&priBuf, pointsMramAligned, sizeof(uint64_t));
        mutex_unlock(mutex_unaligned_neighbors);
        ++pointsMramAligned;
        wramWriteBufSize -= priCopiedBytes;
        memmove(wramWriteBuf, wramWriteBuf + priCopiedBytes, wramWriteBufSize);
        __dma_aligned uint8_t *wramWriteBufPt = wramWriteBuf;
        while ((pqTop = pqueue_pop(pq)) != NULL) {
            memcpy(wramWriteBufPt, pqTop, sizeof(pqueue_elem_t_mram));
            increAddr(wramWriteBufPt);
        }
        mutex_lock(mutex_unaligned_neighbors);
        mram_write_unaligned(wramWriteBuf, pointsMramAligned, wramWriteBufPt - wramWriteBuf);
        mutex_unlock(mutex_unaligned_neighbors);
        fsb_free(wramWriteBufAllocator, wramWriteBuf);
    } else {
        __mram_ptr pqueue_elem_t_mram *pointsMramAddr = pointsMram + pointNeighborStartAddr;
        __dma_aligned pqueue_elem_t *pqTop = NULL;
        while ((pqTop = pqueue_pop(pq)) != NULL) {
            mram_write(pqTop, pointsMramAddr, sizeof(pqueue_elem_t_mram));
            ++pointsMramAddr;
        }
    }
}

void graphBuilding(const __mram_ptr ELEMTYPE *const points, const ADDRTYPE pointAmt, const unsigned short dimAmt, const uint32_t neighborAmt, const ADDRTYPE pointNeighborStartAddr, __mram_ptr pqueue_elem_t_mram *neighbors) {
    fsb_allocator_t pq_allocator;
    pqueue_t *pq;
    fsb_allocator_t pqElemsAllocator = fsb_alloc(neighborAmt * sizeof(pqueue_elem_t), 1);
    pqueue_elem_t *pqElems = (pqueue_elem_t *)fsb_get(pqElemsAllocator);
    pq = pqueue_init(neighborAmt, cmp_pri, get_pri, set_pri, get_pos, set_pos, &pq_allocator);
    uint32_t pointSize = sizeof(ELEMTYPE) * dimAmt;
    const __mram_ptr ELEMTYPE *const pointBorder = points + pointAmt;
    const __mram_ptr ELEMTYPE *const pointReadBorder = points + pointAmt * pointSize;
    fsb_allocator_t curPointBufAllocator = fsb_alloc(pointSize, 1);
    __dma_aligned ELEMTYPE *curPointBuf = fsb_get(curPointBufAllocator);
    seqreader_t curPointSR;
    uint8_t *curPointCache = seqread_init(seqread_alloc(), (__mram_ptr ELEMTYPE *)points + me() * dimAmt, &curPointSR);
    fsb_allocator_t leafPointBufAllocator = fsb_alloc(pointSize, 1);
    __dma_aligned ELEMTYPE *leafPointBuf = fsb_get(leafPointBufAllocator);
    seqreader_t leafPointSR;
    uint8_t *leafPointCache = seqread_init(seqread_alloc(), (__mram_ptr ELEMTYPE *)points, &leafPointSR);
    __mram_ptr pqueue_elem_t_mram *neighborsWrite = neighbors + me() * neighborAmt;
    uint32_t neighborsWriteStride = neighborAmt * NR_TASKLETS;
    for (__mram_ptr ELEMTYPE *curPointPt = (__mram_ptr ELEMTYPE *)points + me(); curPointPt < pointBorder; curPointPt += NR_TASKLETS) {
        for (uint32_t readBytes = 0; readBytes < pointSize; readBytes += SEQREAD_CACHE_SIZE) {
            if (pointSize > SEQREAD_CACHE_SIZE) {
                uint32_t curReadBytes = pointSize - readBytes;
                if (curReadBytes > SEQREAD_CACHE_SIZE)
                    curReadBytes = SEQREAD_CACHE_SIZE;
                memcpy(curPointBuf + readBytes, curPointCache, curReadBytes);
                __mram_ptr uint8_t *seqreadPrePt = (__mram_ptr uint8_t *)seqread_tell(curPointCache, &curPointSR);
                seqread_get(curPointCache, curReadBytes, &curPointSR);  // Note: the official document says this function `consists in incrementing the pointer`, but it seems that it does not here.
                                                                        //       So adjust the pointer with tell/seek manually.
                                                                        //       It does not mind if it `consists in incrementing the pointer` really since the told pointer is the previous one
                seqread_seek(seqreadPrePt + curReadBytes, &curPointSR);
            } else {
                memcpy(curPointBuf + readBytes, curPointCache, pointSize);
                __mram_ptr uint8_t *seqreadPrePt = (__mram_ptr uint8_t *)seqread_tell(curPointCache, &curPointSR);
                seqread_get(curPointCache, pointSize, &curPointSR);
                seqread_seek(seqreadPrePt + pointSize, &curPointSR);
            }
        }
        uint32_t pqElemSize = 0;
        for (__mram_ptr ELEMTYPE *leafPointPt = (__mram_ptr ELEMTYPE *)points; leafPointPt < pointBorder; ++leafPointPt) {
            if (curPointPt != leafPointPt) {
                for (uint32_t readBytes = 0; readBytes < pointSize; readBytes += SEQREAD_CACHE_SIZE) {
                    if (pointSize > SEQREAD_CACHE_SIZE) {
                        uint32_t leafReadBytes = pointSize - readBytes;
                        if (leafReadBytes > SEQREAD_CACHE_SIZE)
                            leafReadBytes = SEQREAD_CACHE_SIZE;
                        memcpy(leafPointBuf + readBytes, leafPointCache, leafReadBytes);
                        __mram_ptr uint8_t *seqreadPrePt = (__mram_ptr uint8_t *)seqread_tell(leafPointCache, &leafPointSR);
                        seqread_get(leafPointCache, leafReadBytes, &leafPointSR);
                        seqread_seek(seqreadPrePt + leafReadBytes, &leafPointSR);
                    } else {
                        memcpy(leafPointBuf + readBytes, leafPointCache, pointSize);
                        __mram_ptr uint8_t *seqreadPrePt = (__mram_ptr uint8_t *)seqread_tell(leafPointCache, &leafPointSR);
                        seqread_get(leafPointCache, pointSize, &leafPointSR);
                        seqread_seek(seqreadPrePt + pointSize, &leafPointSR);
                    }
                }
                pqueue_pri_t dist = distCalVec(curPointBuf, leafPointBuf, dimAmt);
                if (pqElemSize < neighborAmt) {
                    pqElems[pqElemSize].pri = dist, pqElems[pqElemSize].val = leafPointPt - points;
                    pqueue_insert(pq, pqElems + pqElemSize);
                    ++pqElemSize;
                } else {
                    pqueue_elem_t *pqTop = pqueue_peek(pq);
                    if (pqTop->pri > dist) {
                        pqTop->val = leafPointPt - points;
                        pqTop->pri = dist;
                        pqueue_pop(pq);
                        pqueue_insert(pq, pqTop);
                    }
                }
            } else {
                seqread_seek((__mram_ptr uint8_t *)seqread_tell(leafPointCache, &leafPointSR) + pointSize, &leafPointSR);
            }
        }
        seqread_seek((__mram_ptr uint8_t *)points, &leafPointSR);
        save_pq_into_mram(pq, pointNeighborStartAddr, neighborsWrite);  // Pop all elements in pq here! Leave redundant space for incremental updating
        neighborsWrite += neighborsWriteStride;
        __mram_ptr uint8_t *curPointNextReadPt = (__mram_ptr uint8_t *)seqread_tell(curPointCache, &curPointSR) + (NR_TASKLETS - 1) * pointSize;
        if (curPointNextReadPt < (__mram_ptr uint8_t *)pointReadBorder)
            seqread_seek(curPointNextReadPt, &curPointSR);
    }
    fsb_free(leafPointBufAllocator, leafPointBuf);
    fsb_free(curPointBufAllocator, curPointBuf);
    pqueue_free(pq, &pq_allocator);
    fsb_free(pqElemsAllocator, pqElems);
}
