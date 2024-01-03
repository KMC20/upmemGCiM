/*
Author: KMC20
Date: 2023/12/4
Function: Management including data transfer with DPU and for the building phases of GCiM.
*/

#define _GNU_SOURCE
#include <math.h>  // ceil
#include <time.h>
#include <string.h>  // memmove
#include <getopt.h>  // getopt
#include <stdlib.h>  // EXIT_SUCCESS, EXIT_FAILURE; srand
#include <stdio.h>  // FILE
#include <unistd.h>  // access
#include <errno.h>  // errno
#include <dpu.h>
#include <sys/time.h>  // gettimeofday
#include <dpu_error.h>  // dpu_error_t (and DPU_OK, etc)
#include "request.h"
#ifdef ENERGY_EVAL
#include "measureEnergy.h"
#endif

#ifdef PERF_EVAL
typedef uint64_t perfcounter_t;
#define max(a, b) a > b ? a : b
#endif

#define XSTR(x) #x
#define STR(x) XSTR(x)

#define LARGE_TREE_THRESHOLD (32 << 20)  // This value should be in [leaf capacity, Mram size]
#define min(a, b) a < b ? a : b

DPU_INCBIN(dpu_binary_TBP_accumulator, DPU_BINARY_TBP_ACCUMULATOR)
DPU_INCBIN(dpu_binary_TBP_meanSpliter, DPU_BINARY_TBP_MEANSPLITER)
DPU_INCBIN(dpu_binary_TBP, DPU_BINARY_TBP)
DPU_INCBIN(dpu_binary_GBP, DPU_BINARY_GBP)


ADDRTYPE getPointsAmount(const char *const pointsFileName, const uint32_t dimAmt) {
    FILE *fp = fopen(pointsFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input point file: %s! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    ADDRTYPE pointsAmt = ftell(fp) / (sizeof(ELEMTYPE) * dimAmt);
    fclose(fp);
    return pointsAmt;
}

void loadPointsFromFile(const char *const pointsFileName, ELEMTYPE *points) {
    FILE *fp = fopen(pointsFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input point file: %s! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    long long int pointsElemSize = ftell(fp) / sizeof(ELEMTYPE);
    fseek(fp, 0, SEEK_SET);
    if (fread(points, sizeof(ELEMTYPE), pointsElemSize, fp) == 0) {
        fclose(fp);
        printf("The input point file: %s is an empty file! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fclose(fp);
}

void saveDataToFile(const char *const dataFileName, const void *data, const size_t size, const size_t nmemb) {
    FILE *fp = fopen(dataFileName, "wb");
    if (fp == NULL) {
        printf("Failed to open the output point file: %s! Exit now!\n", dataFileName);
        exit(-1);
    }
    fwrite(data, size, nmemb, fp);
    fclose(fp);
}

typedef struct {
    uint64_t pointAmt;
    ELEMTYPE *points;
    uint32_t *dpu_offset;
    ADDRTYPE *pointSizes;
    uint32_t dimAmt;
    uint32_t nr_ranks;
} loadPointsIntoDPUsContext;
dpu_error_t loadPointsIntoDPUs(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    loadPointsIntoDPUsContext *ctx = (loadPointsIntoDPUsContext *)args;
    ELEMTYPE *points = ctx->points;
    uint32_t *dpu_offset = ctx->dpu_offset;
    ADDRTYPE *pointSizes = ctx->pointSizes;
    uint32_t dimAmt = ctx->dimAmt;
    uint64_t pointAmt = ctx->pointAmt;
    uint32_t nr_ranks = ctx->nr_ranks;

    uint64_t pointAmtPerDPU = ceil((double)pointAmt / dpu_offset[nr_ranks]);
    uint64_t elementPerDPU = pointAmtPerDPU * dimAmt;
    unsigned int each_dpu;
    struct dpu_set_t dpu;
    // DPU_FOREACH (rank, dpu, each_dpu) {
    //     DPU_ASSERT(dpu_prepare_xfer(dpu, &points[(each_dpu + dpu_offset[rank_id]) * elementPerDPU]));
    // }
    // DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, sizeof(ELEMTYPE) * elementPerDPU, DPU_XFER_DEFAULT));  // Use `DPU_MRAM_HEAP_POINTER_NAME` for reuse
    DPU_FOREACH (rank, dpu, each_dpu) {
        ADDRTYPE pointSize = pointSizes[each_dpu + dpu_offset[rank_id]] * dimAmt * sizeof(ELEMTYPE);
        if (pointSize > 0)
            DPU_ASSERT(dpu_copy_to(dpu, "points", 0, &points[(each_dpu + dpu_offset[rank_id]) * elementPerDPU], pointSize));
        DPU_ASSERT(dpu_prepare_xfer(dpu, &pointSizes[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, STR(pointAmt), 0, sizeof(ADDRTYPE), DPU_XFER_DEFAULT));

    return DPU_OK;
}

typedef struct {
    MEAN_VALUE_TYPE *sums;
    uint32_t *dpu_offset;
#ifdef PERF_EVAL_SIM
    perfcounter_t *perfs;
    uint32_t *freqs;
#endif
} appendSumToDPUsContext;
dpu_error_t appendSumToDPUs(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    appendSumToDPUsContext *ctx = (appendSumToDPUsContext *)args;
    MEAN_VALUE_TYPE *sums = ctx->sums;
    uint32_t *dpu_offset = ctx->dpu_offset;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &sums[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "sumRes", 0, sizeof(MEAN_VALUE_TYPE), DPU_XFER_DEFAULT));

    return DPU_OK;
}
#ifdef PERF_EVAL_SIM
dpu_error_t getPerfResponseFromMeanCal(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    appendSumToDPUsContext *ctx = (appendSumToDPUsContext *)args;
    uint32_t *dpu_offset = ctx->dpu_offset;
    perfcounter_t *perfs = ctx->perfs;
    uint32_t *freqs = ctx->freqs;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &perfs[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "exec_time", 0, sizeof(perfcounter_t), DPU_XFER_DEFAULT));
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &freqs[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "CLOCKS_PER_SEC", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));

    return DPU_OK;
}
#endif

typedef struct {
    ADDRTYPE *pointSizes;
    ADDRTYPE *splits;
    ELEMTYPE **iterPoints;
    uint32_t *dpu_offset;
    uint32_t dimAmt;
    uint32_t max_dpus;
#ifdef PERF_EVAL_SIM
    perfcounter_t *perfs;
    uint32_t *freqs;
#endif
} getResponseFromLargeTreesContext;
dpu_error_t getResponseFromLargeTreesPart1(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromLargeTreesContext *ctx = (getResponseFromLargeTreesContext *)args;
    uint32_t *dpu_offset = ctx->dpu_offset;
    ADDRTYPE *splits = ctx->splits;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &splits[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "splitRes", 0, sizeof(ADDRTYPE), DPU_XFER_DEFAULT));

    return DPU_OK;
}
dpu_error_t getResponseFromLargeTreesPart2(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromLargeTreesContext *ctx = (getResponseFromLargeTreesContext *)args;
    uint32_t *dpu_offset = ctx->dpu_offset;
    uint32_t dimAmt = ctx->dimAmt;
    uint32_t max_dpus = ctx->max_dpus;
    ADDRTYPE *pointSizes = ctx->pointSizes;
    ADDRTYPE *splits = ctx->splits;
    ELEMTYPE **iterPoints = ctx->iterPoints;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu >= max_dpus)
            continue;
        ADDRTYPE lsplit = splits[nr_dpu] * dimAmt;
        // DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, (uint8_t *)iterPoints[nr_dpu], sizeof(ELEMTYPE) * lsplit));
        if (lsplit > 0)  // It is possible that all points distributed to a dpu should be split into the left/right child node during the construction phase of the top tree, and it is invalid to transfer no data with the dpu API
            DPU_ASSERT(dpu_copy_from(dpu, "points", 0, (uint8_t *)iterPoints[nr_dpu], sizeof(ELEMTYPE) * lsplit));
    }
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu >= max_dpus)
            continue;
        ADDRTYPE rsplit = (pointSizes[nr_dpu] - splits[nr_dpu]) * dimAmt;
        // DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, sizeof(ELEMTYPE) * splits[nr_dpu] * dimAmt, (uint8_t *)iterPoints[nr_dpu + nr_all_dpus], sizeof(ELEMTYPE) * rsplit));
        if (rsplit > 0)  // It is possible that all points distributed to a dpu should be split into the left/right child node during the construction phase of the top tree, and it is invalid to transfer no data with the dpu API
            DPU_ASSERT(dpu_copy_from(dpu, "points", sizeof(ELEMTYPE) * splits[nr_dpu] * dimAmt, (uint8_t *)iterPoints[nr_dpu + max_dpus], sizeof(ELEMTYPE) * rsplit));
    }

    return DPU_OK;
}
#ifdef PERF_EVAL_SIM
dpu_error_t getPerfResponseFromMeanSpliter(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromLargeTreesContext *ctx = (getResponseFromLargeTreesContext *)args;
    uint32_t *dpu_offset = ctx->dpu_offset;
    perfcounter_t *perfs = ctx->perfs;
    uint32_t *freqs = ctx->freqs;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &perfs[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "exec_time", 0, sizeof(perfcounter_t), DPU_XFER_DEFAULT));
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &freqs[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "CLOCKS_PER_SEC", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));

    return DPU_OK;
}
#endif

typedef struct {
    ADDRTYPE max_dpus;
    ELEMTYPE *points;
    uint32_t *dpu_offset;
    ADDRTYPE *treeLeftAddr;
    ADDRTYPE *treeSize;
    ADDRTYPE *leafIds;
    uint32_t dimAmt;
} loadLargeLeavesIntoDPUsContext;
dpu_error_t loadLargeLeavesIntoDPUs(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    loadLargeLeavesIntoDPUsContext *ctx = (loadLargeLeavesIntoDPUsContext *)args;
    ELEMTYPE *points = ctx->points;
    uint32_t *dpu_offset = ctx->dpu_offset;
    ADDRTYPE *treeLeftAddr = ctx->treeLeftAddr;
    ADDRTYPE *treeSize = ctx->treeSize;
    ADDRTYPE *leafIds = ctx->leafIds;
    uint32_t dimAmt = ctx->dimAmt;
    ADDRTYPE max_dpus = ctx->max_dpus;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu >= max_dpus)
            break;
        DPU_ASSERT(dpu_copy_to(dpu, "points", 0, (uint8_t *)&points[treeLeftAddr[leafIds[nr_dpu]] * dimAmt], sizeof(ELEMTYPE) * treeSize[leafIds[nr_dpu]] * dimAmt));
    }
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu < max_dpus)
            DPU_ASSERT(dpu_prepare_xfer(dpu, &treeSize[leafIds[nr_dpu]]));
        else
            DPU_ASSERT(dpu_prepare_xfer(dpu, &treeSize[leafIds[max_dpus - 1]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, "pointAmt", 0, sizeof(ADDRTYPE), DPU_XFER_DEFAULT));

    return DPU_OK;
}

typedef struct {
    ADDRTYPE max_dpus;
    ADDRTYPE *subtreeSizes;
    ADDRTYPE *treeIdSizes;
    ADDRTYPE *newLeafSizes;
    ADDRTYPE *leafIdAddrs;
    ELEMTYPE *points;
    uint32_t *dpu_offset;
    treeNode_t *tree;
    ADDRTYPE *treeLeftAddr;
    ADDRTYPE *treeSize;
    ADDRTYPE *leafIds;
    uint32_t dimAmt;
#ifdef PERF_EVAL_SIM
    perfcounter_t *perfs;
    uint32_t *freqs;
#endif
} getResponseFromTreesContext;
dpu_error_t getResponseFromTreesPart1(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromTreesContext *ctx = (getResponseFromTreesContext *)args;
    ELEMTYPE *points = ctx->points;
    uint32_t *dpu_offset = ctx->dpu_offset;
    ADDRTYPE *treeLeftAddr = ctx->treeLeftAddr;
    ADDRTYPE *treeSize = ctx->treeSize;
    ADDRTYPE *leafIds = ctx->leafIds;
    uint32_t dimAmt = ctx->dimAmt;
    ADDRTYPE max_dpus = ctx->max_dpus;
    ADDRTYPE *subtreeSizes = ctx->subtreeSizes;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu >= max_dpus)
            break;
        DPU_ASSERT(dpu_copy_from(dpu, "points", 0, (uint8_t *)&points[treeLeftAddr[leafIds[nr_dpu]] * dimAmt], sizeof(ELEMTYPE) * treeSize[leafIds[nr_dpu]] * dimAmt));
        DPU_ASSERT(dpu_copy_from(dpu, "treeSizeRes", 0, (uint8_t *)&subtreeSizes[nr_dpu], sizeof(ADDRTYPE)));
    }

    return DPU_OK;
}
dpu_error_t getResponseFromTreesPart2(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromTreesContext *ctx = (getResponseFromTreesContext *)args;
    uint32_t *dpu_offset = ctx->dpu_offset;
    treeNode_t *tree = ctx->tree;
    ADDRTYPE *treeLeftAddr = ctx->treeLeftAddr;
    ADDRTYPE *leafIds = ctx->leafIds;
    ADDRTYPE max_dpus = ctx->max_dpus;
    ADDRTYPE *treeIdSizes = ctx->treeIdSizes;
    ADDRTYPE *subtreeSizes = ctx->subtreeSizes;
    ADDRTYPE *newLeafSizes = ctx->newLeafSizes;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu >= max_dpus)
            break;
        DPU_ASSERT(dpu_copy_from(dpu, "tree", 0, (uint8_t *)&tree[leafIds[nr_dpu]], sizeof(treeNode_t)));
        if (subtreeSizes[nr_dpu] > 0) {
            DPU_ASSERT(dpu_copy_from(dpu, "tree", sizeof(treeNode_t), (uint8_t *)&tree[treeIdSizes[nr_dpu]], sizeof(treeNode_t) * subtreeSizes[nr_dpu]));
        }

        newLeafSizes[nr_dpu] = 0;
        if (tree[leafIds[nr_dpu]].left != (ADDRTYPE)(uint64_t)NULL || tree[leafIds[nr_dpu]].right != (ADDRTYPE)(uint64_t)NULL) {
            if (tree[leafIds[nr_dpu]].left != (ADDRTYPE)(uint64_t)NULL)
                tree[leafIds[nr_dpu]].left += treeIdSizes[nr_dpu] - 1;
            if (tree[leafIds[nr_dpu]].right != (ADDRTYPE)(uint64_t)NULL)
                tree[leafIds[nr_dpu]].right += treeIdSizes[nr_dpu] - 1;
        } else {
            tree[leafIds[nr_dpu]].mean += treeLeftAddr[leafIds[nr_dpu]];
            ++(newLeafSizes[nr_dpu]);
        }
        ADDRTYPE treeIdSizesEnd = treeIdSizes[nr_dpu] + subtreeSizes[nr_dpu];
        for (ADDRTYPE treeIdSize = treeIdSizes[nr_dpu]; treeIdSize < treeIdSizesEnd; ++treeIdSize) {
            if (tree[treeIdSize].left != (ADDRTYPE)(uint64_t)NULL || tree[treeIdSize].right != (ADDRTYPE)(uint64_t)NULL) {
                if (tree[treeIdSize].left != (ADDRTYPE)(uint64_t)NULL)
                    tree[treeIdSize].left += treeIdSizes[nr_dpu] - 1;
                if (tree[treeIdSize].right != (ADDRTYPE)(uint64_t)NULL)
                    tree[treeIdSize].right += treeIdSizes[nr_dpu] - 1;
            } else {
                tree[treeIdSize].mean += treeLeftAddr[leafIds[nr_dpu]];
                ++(newLeafSizes[nr_dpu]);
            }
        }
    }

    return DPU_OK;
}
dpu_error_t getResponseFromTreesPart3(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromTreesContext *ctx = (getResponseFromTreesContext *)args;
    uint32_t *dpu_offset = ctx->dpu_offset;
    treeNode_t *tree = ctx->tree;
    ADDRTYPE *leafIds = ctx->leafIds;
    ADDRTYPE max_dpus = ctx->max_dpus;
    ADDRTYPE *treeIdSizes = ctx->treeIdSizes;
    ADDRTYPE *leafIdAddrs = ctx->leafIdAddrs;
    ADDRTYPE *subtreeSizes = ctx->subtreeSizes;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu >= max_dpus || &dpu == &rank)  // `&dpu == &rank` is just used to avoid warnings. It is expected to be false
            break;

        ADDRTYPE leafIdAddr = leafIdAddrs[nr_dpu];
        if (tree[leafIds[nr_dpu]].left == (ADDRTYPE)(uint64_t)NULL && tree[leafIds[nr_dpu]].right == (ADDRTYPE)(uint64_t)NULL) {
            leafIds[leafIdAddr++] = leafIds[nr_dpu];
        }
        ADDRTYPE treeIdSizesEnd = treeIdSizes[nr_dpu] + subtreeSizes[nr_dpu];
        for (ADDRTYPE treeIdSize = treeIdSizes[nr_dpu]; treeIdSize < treeIdSizesEnd; ++treeIdSize) {
            if (tree[treeIdSize].left == (ADDRTYPE)(uint64_t)NULL && tree[treeIdSize].right == (ADDRTYPE)(uint64_t)NULL) {
                leafIds[leafIdAddr++] = treeIdSize;
            }
        }
    }

    return DPU_OK;
}
#ifdef PERF_EVAL_SIM
dpu_error_t getPerfResponseFromTrees(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromTreesContext *ctx = (getResponseFromTreesContext *)args;
    uint32_t *dpu_offset = ctx->dpu_offset;
    perfcounter_t *perfs = ctx->perfs;
    uint32_t *freqs = ctx->freqs;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &perfs[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "exec_time", 0, sizeof(perfcounter_t), DPU_XFER_DEFAULT));
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &freqs[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "CLOCKS_PER_SEC", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));

    return DPU_OK;
}
#endif

typedef struct {
    ADDRTYPE max_dpus;
    ELEMTYPE *points;
    uint32_t *dpu_offset;
    treeNode_t *tree;
    ADDRTYPE *leafIds;
    uint32_t dimAmt;
} loadLeavesIntoDPUsContext;
dpu_error_t loadLeavesIntoDPUs(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    loadLeavesIntoDPUsContext *ctx = (loadLeavesIntoDPUsContext *)args;
    ELEMTYPE *points = ctx->points;
    uint32_t *dpu_offset = ctx->dpu_offset;
    treeNode_t *tree = ctx->tree;
    ADDRTYPE *leafIds = ctx->leafIds;
    uint32_t dimAmt = ctx->dimAmt;
    ADDRTYPE max_dpus = ctx->max_dpus;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu < max_dpus)
            DPU_ASSERT(dpu_prepare_xfer(dpu, &tree[leafIds[nr_dpu]].dim));
        else
            DPU_ASSERT(dpu_prepare_xfer(dpu, &tree[leafIds[max_dpus - 1]].dim));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, "pointAmt", 0, sizeof(ADDRTYPE), DPU_XFER_DEFAULT));
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu >= max_dpus)
            break;
        DPU_ASSERT(dpu_copy_to(dpu, "points", 0, (uint8_t *)&points[tree[leafIds[nr_dpu]].mean * dimAmt], sizeof(ELEMTYPE) * tree[leafIds[nr_dpu]].dim * dimAmt));
    }

    return DPU_OK;
}

typedef struct {
    ADDRTYPE max_dpus;
    pqueue_elem_t_mram *neighbors;
    uint32_t *dpu_offset;
    treeNode_t *tree;
    ADDRTYPE *leafIds;
    uint32_t neighborAmt;
#ifdef PERF_EVAL_SIM
    perfcounter_t *perfs;
    uint32_t *freqs;
#endif
} getResponseFromGraphsContext;
dpu_error_t getResponseFromGraphs(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromGraphsContext *ctx = (getResponseFromGraphsContext *)args;
    pqueue_elem_t_mram *neighbors = ctx->neighbors;
    uint32_t *dpu_offset = ctx->dpu_offset;
    treeNode_t *tree = ctx->tree;
    ADDRTYPE *leafIds = ctx->leafIds;
    uint32_t neighborAmt = ctx->neighborAmt;
    ADDRTYPE max_dpus = ctx->max_dpus;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        unsigned int nr_dpu = each_dpu + dpu_offset[rank_id];
        if (nr_dpu >= max_dpus)
            break;
        DPU_ASSERT(dpu_copy_from(dpu, "neighbors", 0, (uint8_t *)&neighbors[tree[leafIds[nr_dpu]].mean * neighborAmt], sizeof(pqueue_elem_t_mram) * tree[leafIds[nr_dpu]].dim * neighborAmt));  // If the leaf size is smaller than neighborAmt, this operation may cause overflow of address, which leads to a segment fault. Caution please!
    }

    return DPU_OK;
}
#ifdef PERF_EVAL_SIM
dpu_error_t getPerfResponseFromGraphs(struct dpu_set_t rank, uint32_t rank_id, void *args) {
    getResponseFromGraphsContext *ctx = (getResponseFromGraphsContext *)args;
    uint32_t *dpu_offset = ctx->dpu_offset;
    perfcounter_t *perfs = ctx->perfs;
    uint32_t *freqs = ctx->freqs;

    unsigned int each_dpu;
    struct dpu_set_t dpu;
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &perfs[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "exec_time", 0, sizeof(perfcounter_t), DPU_XFER_DEFAULT));
    DPU_FOREACH (rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &freqs[each_dpu + dpu_offset[rank_id]]));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "CLOCKS_PER_SEC", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));

    return DPU_OK;
}
#endif

__attribute__((noreturn)) static void usage(FILE *f, int exit_code, const char *exec_name) {
    /* clang-format off */
    fprintf(f,
#ifdef PERF_EVAL
            "\nusage: %s [-p <points_path>] [-t <tree_result_path>] [-l <leaf_result_path>] [-k <knn_result_path>] [-D <number_of_dimension>] [-F <frequency_of_dpus>] [-K <number_of_neighbors>] [-L <capacity_of_leaves>] [-M <number_of_mrams>]\n"
#else
            "\nusage: %s [-p <points_path>] [-t <tree_result_path>] [-l <leaf_result_path>] [-k <knn_result_path>] [-D <number_of_dimension>] [-K <number_of_neighbors>] [-L <capacity_of_leaves>] [-M <number_of_mrams>]\n"
#endif
            "\n"
            "\t-p \tthe path to the points location (default: points.bin)\n"
            "\t-t \tthe path to the tree result location (default: tree.bin)\n"
            "\t-l \tthe path to the leaf result location (default: leaf.bin)\n"
            "\t-k \tthe path to the k nearest neighbor, a.k.a, the result graph, location (default: knn.bin)\n"
            "\t-D \tthe number of dimensions of input points (default: 128)\n"
#ifdef PERF_EVAL
            "\t-F \tthe frequency of DPUs (default: 450000000)\n"
#endif
            "\t-K \tthe number of neighbors (default: 10)\n"
            "\t-L \tthe capacity of leaves (default: 1000)\n"
            "\t-M \tthe number of mram to used (default: DPU_ALLOCATE_ALL)\n"
            "\t-h \tshow the usage message\n",
            exec_name);
    /* clang-format on */
    exit(exit_code);
}

static void verify_path_exists(const char *path) {
    if (access(path, R_OK)) {
        fprintf(stderr, "path '%s' does not exist or is not readable (errno: %i)\n", path, errno);
        exit(EXIT_FAILURE);
    }
}

#ifdef PERF_EVAL
static void parse_args(int argc, char **argv, uint32_t *dimAmt, uint32_t *neighborAmt, uint32_t *leafCapacity, uint32_t *nb_mram, uint64_t *frequency, char **pointsFileName, char **treeFileName, char **leafFileName, char **knnFileName) {
#else
static void parse_args(int argc, char **argv, uint32_t *dimAmt, uint32_t *neighborAmt, uint32_t *leafCapacity, uint32_t *nb_mram, char **pointsFileName, char **treeFileName, char **leafFileName, char **knnFileName) {
#endif
    int opt;
    extern char *optarg;
#ifdef PERF_EVAL
    while ((opt = getopt(argc, argv, "hD:K:L:M:F:p:t:l:k:")) != -1) {
#else
    while ((opt = getopt(argc, argv, "hD:K:L:M:p:t:l:k:")) != -1) {
#endif
        switch (opt) {
            case 'p':
                *pointsFileName = optarg;
                break;
            case 'D':
                *dimAmt = (uint32_t)atoi(optarg);
                break;
            case 'K':
                *neighborAmt = (uint32_t)atoi(optarg);
                break;
            case 'L':
                *leafCapacity = (uint32_t)atoi(optarg);
                break;
#ifdef PERF_EVAL
            case 'F':
                *frequency = (uint64_t)atoi(optarg);
                break;
#endif
            case 't':
                *treeFileName = optarg;
                break;
            case 'l':
                *leafFileName = optarg;
                break;
            case 'k':
                *knnFileName = optarg;
                break;
            case 'M':
                *nb_mram = (uint32_t)atoi(optarg);
                break;
            case 'h':
                usage(stdout, EXIT_SUCCESS, argv[0]);
            default:
                usage(stderr, EXIT_FAILURE, argv[0]);
        }
    }
    verify_path_exists(*pointsFileName);
}

#ifdef PERF_EVAL
static void allocated_and_compute(struct dpu_set_t dpu_set, uint32_t nr_ranks, const uint32_t dimAmt, const uint32_t neighborAmt, const uint32_t leafCapacity, const char *const pointsFileName, const char *const treeFileName, const char *const leafFileName, const char *const knnFileName) {
#else
static void allocated_and_compute(struct dpu_set_t dpu_set, uint32_t nr_ranks, const uint32_t dimAmt, const uint32_t neighborAmt, const uint32_t leafCapacity, const char *const pointsFileName, const char *const treeFileName, const char *const leafFileName, const char *const knnFileName) {
#endif
    const ADDRTYPE pointAmt = getPointsAmount(pointsFileName, dimAmt);
    const ADDRTYPE MAX_TREE_SIZE = pointAmt + 3;
    ELEMTYPE *points = malloc(pointAmt * dimAmt * sizeof(ELEMTYPE));
    loadPointsFromFile(pointsFileName, points);

#ifdef ENERGY_EVAL
    double ESU = getEnergyUnit();
    uint32_t nr_sockets = getNRSockets();
    uint32_t nr_cpus = getNRPhyCPUs();
    uint32_t evalCPUIds[nr_sockets];
    for (uint32_t nr_cpu = 0, nr_socket = 0, coresPerSocket = nr_cpus / nr_sockets; nr_socket < nr_sockets; nr_cpu += coresPerSocket, ++nr_socket)
        evalCPUIds[nr_socket] = nr_cpu;
    uint64_t totalExecEnergy = 0;
    uint64_t startEnergy[nr_sockets], endEnergy[nr_sockets];
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
#ifdef PERF_EVAL
    uint64_t totalExecTime = 0;  // Unit: us
    uint64_t dpuExecTime = 0, hostExecTime = 0, dataTransferhost2DPUTime = 0, dataTransferDPU2hostTime = 0;  // Unit: us; the sum of these may exceed the total time since the time for responses and dpu execution may be overlapped
    uint64_t TBPExecTime = 0, GBPExecTime = 0;  // Unit: us
#endif
#if (defined PRINT_PERF_EACH_PHASE || defined PERF_EVAL)
    long start, end;
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif

    // Set dpu_offset
    uint32_t dpu_offset[nr_ranks + 1];
    dpu_offset[0] = 0;
    uint32_t nr_all_dpus = 0;

    struct dpu_set_t rank;
    uint32_t each_rank;
    DPU_RANK_FOREACH (dpu_set, rank, each_rank) {
        uint32_t nr_dpus;
        DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus));
        dpu_offset[each_rank + 1] = dpu_offset[each_rank] + nr_dpus;
        nr_all_dpus += nr_dpus;
    }

#ifdef PERF_EVAL
    gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    totalExecTime += end - start;
    hostExecTime += end - start;
    TBPExecTime = totalExecTime;
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
    // printf("Initializing buffers\n");
    treeNode_t *tree = malloc(MAX_TREE_SIZE * sizeof(treeNode_t));
    ADDRTYPE *treeLeftAddr = malloc(MAX_TREE_SIZE * sizeof(ADDRTYPE));
    ADDRTYPE *treeSize = malloc(MAX_TREE_SIZE * sizeof(ADDRTYPE));
    ADDRTYPE *largeTreeIds = malloc(pointAmt * dimAmt * sizeof(treeNode_t) / LARGE_TREE_THRESHOLD * sizeof(ADDRTYPE));
    ADDRTYPE largeTreeIdSize = 0;
    ADDRTYPE treeIdSize = 0;
    tree[0].left = tree[0].right = (ADDRTYPE)(uint64_t)NULL;
    treeLeftAddr[0] = 0;
    treeSize[0] = pointAmt;
    tree[0].mean = treeLeftAddr[0], tree[0].dim = treeSize[0];
    ++treeIdSize;
    if (pointAmt * dimAmt * sizeof(treeNode_t) > LARGE_TREE_THRESHOLD)
        largeTreeIds[largeTreeIdSize++] = 0;
    srand(time(0));
    // 1. Transfer data to DPU
    // 2. TBP (here, I record the left most address of points in each corresponding leaf node to reduce memory usage, which might be changed into `leafId * leafCapacity` for the future incremental updating)
    // printf("Tree building phase:\n");
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time for data preparation: %.3lfs\n", (end - start) / 1e6);
#endif
    // Split all large subtrees
    while (largeTreeIdSize > 0) {
        MEAN_VALUE_TYPE sums[nr_all_dpus];
        ADDRTYPE pointSizes[nr_all_dpus];
        ADDRTYPE splits[nr_all_dpus];
        uint32_t iterPointsSize = nr_all_dpus << 1;
        ELEMTYPE *iterPoints[iterPointsSize];
        ADDRTYPE newLargeTreeIdSize = 0;
        for (ADDRTYPE largeTreeId = 0; largeTreeId < largeTreeIdSize; ++largeTreeId) {
            ADDRTYPE max_dpus = nr_all_dpus;
            DPU_ASSERT(dpu_load_from_incbin(dpu_set, &dpu_binary_TBP_accumulator, NULL));
            // Send data to DPUs
            loadPointsIntoDPUsContext loadPointsIntoDPUsContext_ctx = { .pointAmt = treeSize[largeTreeIds[largeTreeId]], .points = points + treeLeftAddr[largeTreeIds[largeTreeId]], .dpu_offset = dpu_offset, .pointSizes = pointSizes, .dimAmt = dimAmt, .nr_ranks = nr_ranks };
            {
                uint64_t pointAmtPerDPU = ceil((double)loadPointsIntoDPUsContext_ctx.pointAmt / dpu_offset[nr_ranks]);
                uint64_t pointCnt = 0;
                for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
                    pointSizes[nr_dpu] = min(loadPointsIntoDPUsContext_ctx.pointAmt - pointCnt, pointAmtPerDPU);
                    pointCnt += pointSizes[nr_dpu];
                    if (pointSizes[nr_dpu] < 1) {
                        max_dpus = nr_dpu;
                        break;
                    }
                }
            }
            uint32_t dim = rand() % dimAmt;
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            hostExecTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(dim), 0, &dim, sizeof(uint32_t), DPU_XFER_ASYNC));
            DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(dimAmt), 0, &dimAmt, sizeof(uint32_t), DPU_XFER_ASYNC));
            DPU_ASSERT(dpu_callback(dpu_set, loadPointsIntoDPUs, &loadPointsIntoDPUsContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            dataTransferhost2DPUTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            dpuExecTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
#ifdef PERF_EVAL_SIM
            perfcounter_t perfs[nr_all_dpus];
            uint32_t freqs[nr_all_dpus];
            appendSumToDPUsContext appendSumToDPUsContext_ctx = { .sums = sums, .dpu_offset = dpu_offset, .perfs = perfs, .freqs = freqs };
#else
            appendSumToDPUsContext appendSumToDPUsContext_ctx = { .sums = sums, .dpu_offset = dpu_offset };
#endif
            DPU_ASSERT(dpu_callback(dpu_set, appendSumToDPUs, &appendSumToDPUsContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            dataTransferDPU2hostTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            DPU_ASSERT(dpu_load_from_incbin(dpu_set, &dpu_binary_TBP_meanSpliter, NULL));
            MEAN_VALUE_TYPE splitVal = 0;
            for (ADDRTYPE nr_dpu = 0; nr_dpu < max_dpus; ++nr_dpu)
                splitVal += sums[nr_dpu];
            splitVal /= treeSize[largeTreeIds[largeTreeId]];
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            hostExecTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(mean), 0, &splitVal, sizeof(MEAN_VALUE_TYPE), DPU_XFER_ASYNC));
            DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(dim), 0, &dim, sizeof(uint32_t), DPU_XFER_ASYNC));
            DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(dimAmt), 0, &dimAmt, sizeof(uint32_t), DPU_XFER_ASYNC));
            // It seems that DPUs would not preserve variables after loading new binary codes, so transfer the data again
            DPU_ASSERT(dpu_callback(dpu_set, loadPointsIntoDPUs, &loadPointsIntoDPUsContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            dataTransferhost2DPUTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            dpuExecTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            // Get responses and update largeTreeIds, tree, treeSize, largeTreeIdSize and newLargeTreeIdSize
#ifdef PERF_EVAL_SIM
            getResponseFromLargeTreesContext getResponseFromLargeTreesContext_ctx = { .pointSizes = pointSizes, .splits = splits, .iterPoints = iterPoints, .dpu_offset = dpu_offset, .dimAmt = dimAmt, .max_dpus = max_dpus, .perfs = perfs, .freqs = freqs };
#else
            getResponseFromLargeTreesContext getResponseFromLargeTreesContext_ctx = { .pointSizes = pointSizes, .splits = splits, .iterPoints = iterPoints, .dpu_offset = dpu_offset, .dimAmt = dimAmt, .max_dpus = max_dpus };
#endif
            DPU_ASSERT(dpu_callback(dpu_set, getResponseFromLargeTreesPart1, &getResponseFromLargeTreesContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            dataTransferDPU2hostTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            {
                iterPoints[0] = loadPointsIntoDPUsContext_ctx.points;
                for (uint32_t iterPointsCnt = 1; iterPointsCnt < max_dpus; ++iterPointsCnt)
                    iterPoints[iterPointsCnt] = iterPoints[iterPointsCnt - 1] + getResponseFromLargeTreesContext_ctx.splits[iterPointsCnt - 1] * dimAmt;
                if (max_dpus > 1)
                    iterPoints[max_dpus] = iterPoints[max_dpus - 1] + getResponseFromLargeTreesContext_ctx.splits[max_dpus - 1] * dimAmt;
                for (uint32_t iterPointsCnt = max_dpus + 1, splitCnt = 0; iterPointsCnt < iterPointsSize; ++iterPointsCnt, ++splitCnt)
                    iterPoints[iterPointsCnt] = iterPoints[iterPointsCnt - 1] + (getResponseFromLargeTreesContext_ctx.pointSizes[splitCnt] - getResponseFromLargeTreesContext_ctx.splits[splitCnt]) * dimAmt;
            }
            ADDRTYPE leftPointSize = (iterPoints[max_dpus] - loadPointsIntoDPUsContext_ctx.points) / dimAmt;
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            hostExecTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            DPU_ASSERT(dpu_callback(dpu_set, getResponseFromLargeTreesPart2, &getResponseFromLargeTreesContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
            gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
            end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
            totalExecTime += end - start;
            dataTransferDPU2hostTime += end - start;
#ifdef ENERGY_EVAL
            for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
                startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
            gettimeofday(&timecheck, NULL);
            start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
            // Update tree data structures
            tree[largeTreeIds[largeTreeId]].mean = splitVal, tree[largeTreeIds[largeTreeId]].dim = dim;
            if (leftPointSize > 0) {
                tree[largeTreeIds[largeTreeId]].left = treeIdSize;
                tree[treeIdSize].left = tree[treeIdSize].right = (ADDRTYPE)(uint64_t)NULL;
                treeLeftAddr[treeIdSize] = treeLeftAddr[largeTreeIds[largeTreeId]];
                treeSize[treeIdSize] = leftPointSize;
                tree[treeIdSize].mean = treeLeftAddr[treeIdSize], tree[treeIdSize].dim = treeSize[treeIdSize];
                if (leftPointSize * dimAmt * sizeof(treeNode_t) > LARGE_TREE_THRESHOLD)
                    largeTreeIds[largeTreeIdSize + newLargeTreeIdSize++] = treeIdSize;
                ++treeIdSize;
            }
            ADDRTYPE rightPointSize = treeSize[largeTreeIds[largeTreeId]] - leftPointSize;
            if (rightPointSize > 0) {
                tree[largeTreeIds[largeTreeId]].right = treeIdSize;
                tree[treeIdSize].left = tree[treeIdSize].right = (ADDRTYPE)(uint64_t)NULL;
                treeLeftAddr[treeIdSize] = treeLeftAddr[largeTreeIds[largeTreeId]] + leftPointSize;
                treeSize[treeIdSize] = rightPointSize;
                tree[treeIdSize].mean = treeLeftAddr[treeIdSize], tree[treeIdSize].dim = treeSize[treeIdSize];
                if (rightPointSize * dimAmt * sizeof(treeNode_t) > LARGE_TREE_THRESHOLD)
                    largeTreeIds[largeTreeIdSize + newLargeTreeIdSize++] = treeIdSize;
                ++treeIdSize;
            }
        }
        if (newLargeTreeIdSize > 0)
            memmove(largeTreeIds, largeTreeIds + largeTreeIdSize, sizeof(ADDRTYPE) * newLargeTreeIdSize);
        largeTreeIdSize = newLargeTreeIdSize;
    }
    free(largeTreeIds);
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time until top tree building phase completed: %.3lfs\n", (end - start) / 1e6);
#endif
    // Split all subtrees only on DPUs
    ADDRTYPE *leafIds = malloc(MAX_TREE_SIZE * sizeof(ADDRTYPE));
    ADDRTYPE leafIdSize = 0;
    for (ADDRTYPE treeId = 0; treeId < treeIdSize; ++treeId)
        if (tree[treeId].left == (ADDRTYPE)(uint64_t)NULL && tree[treeId].right == (ADDRTYPE)(uint64_t)NULL)
            leafIds[leafIdSize++] = treeId;
    ADDRTYPE largeLeafIdSize = leafIdSize;
    DPU_ASSERT(dpu_sync(dpu_set));
    for (ADDRTYPE TBPbatch = 0; TBPbatch < largeLeafIdSize; TBPbatch += nr_all_dpus) {
        ADDRTYPE subtreeSizes[nr_all_dpus];
        ADDRTYPE treeIdSizes[nr_all_dpus];
        ADDRTYPE newLeafSizes[nr_all_dpus];
        ADDRTYPE leafIdAddrs[nr_all_dpus];
        treeIdSizes[0] = treeIdSize, leafIdAddrs[0] = leafIdSize;
        ADDRTYPE max_dpus = min(largeLeafIdSize - TBPbatch, nr_all_dpus);
        DPU_ASSERT(dpu_load_from_incbin(dpu_set, &dpu_binary_TBP, NULL));
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        hostExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        // Send data to DPUs. Note that redundant DPUs would be ignored
        DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(dimAmt), 0, &dimAmt, sizeof(uint32_t), DPU_XFER_ASYNC));
        DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(leafCapacity), 0, &leafCapacity, sizeof(uint32_t), DPU_XFER_ASYNC));
        loadLargeLeavesIntoDPUsContext loadLargeLeavesIntoDPUsContext_ctx = { .max_dpus = max_dpus, .points = points, .dpu_offset = dpu_offset, .treeLeftAddr = treeLeftAddr, .treeSize = treeSize, .leafIds = leafIds + TBPbatch, .dimAmt = dimAmt };
        DPU_ASSERT(dpu_callback(dpu_set, loadLargeLeavesIntoDPUs, &loadLargeLeavesIntoDPUsContext_ctx, DPU_CALLBACK_DEFAULT));
        // Execute on DPUs
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dataTransferhost2DPUTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));  // Can be replaced by `DPU_ASYNCHRONOUS`. Use `DPU_SYNCHRONOUS` here for measurement of actual DPU executing time. But it seems that the performance with both signs is similar
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dpuExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        // Get responses and update tree and treeSize
#ifdef PERF_EVAL_SIM
        perfcounter_t perfs[nr_all_dpus];
        uint32_t freqs[nr_all_dpus];
        getResponseFromTreesContext getResponseFromTreesContext_ctx = { .max_dpus = max_dpus, .subtreeSizes = subtreeSizes, .treeIdSizes = treeIdSizes, .newLeafSizes = newLeafSizes, .leafIdAddrs = leafIdAddrs, .points = points, .dpu_offset = dpu_offset, .tree = tree, .treeLeftAddr = treeLeftAddr, .treeSize = treeSize, .leafIds = leafIds + TBPbatch, .dimAmt = dimAmt, .perfs = perfs, .freqs = freqs };
#else
        getResponseFromTreesContext getResponseFromTreesContext_ctx = { .max_dpus = max_dpus, .subtreeSizes = subtreeSizes, .treeIdSizes = treeIdSizes, .newLeafSizes = newLeafSizes, .leafIdAddrs = leafIdAddrs, .points = points, .dpu_offset = dpu_offset, .tree = tree, .treeLeftAddr = treeLeftAddr, .treeSize = treeSize, .leafIds = leafIds + TBPbatch, .dimAmt = dimAmt };
#endif
        DPU_ASSERT(dpu_callback(dpu_set, getResponseFromTreesPart1, &getResponseFromTreesContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dataTransferDPU2hostTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        --(subtreeSizes[0]);
        for (uint32_t subtreeSizeCnt = 1; subtreeSizeCnt < max_dpus; ++subtreeSizeCnt) {
            treeIdSizes[subtreeSizeCnt] = treeIdSizes[subtreeSizeCnt - 1] + subtreeSizes[subtreeSizeCnt - 1];
            --(subtreeSizes[subtreeSizeCnt]);
        }
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        hostExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        DPU_ASSERT(dpu_callback(dpu_set, getResponseFromTreesPart2, &getResponseFromTreesContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dataTransferDPU2hostTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        leafIdAddrs[0] -= TBPbatch;
        for (uint32_t leafSizeCnt = 1; leafSizeCnt < max_dpus; ++leafSizeCnt) {
            leafIdAddrs[leafSizeCnt] = leafIdAddrs[leafSizeCnt - 1] + newLeafSizes[leafSizeCnt - 1];
        }
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        hostExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        DPU_ASSERT(dpu_callback(dpu_set, getResponseFromTreesPart3, &getResponseFromTreesContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dataTransferDPU2hostTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        treeIdSize = treeIdSizes[max_dpus - 1] + subtreeSizes[max_dpus - 1], leafIdSize = leafIdAddrs[max_dpus - 1] + TBPbatch + newLeafSizes[max_dpus - 1];  // leafIdAddrs[0] -= TBPbatch;
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        hostExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
    }
    DPU_ASSERT(dpu_sync(dpu_set));
    memmove(leafIds, leafIds + largeLeafIdSize, sizeof(ADDRTYPE) * (leafIdSize - largeLeafIdSize));
    leafIdSize -= largeLeafIdSize;
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time until tree building phase completed: %.3lfs\n", (end - start) / 1e6);
#endif
    free(treeLeftAddr);
    free(treeSize);
#ifdef PERF_EVAL
    gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    totalExecTime += end - start;
    hostExecTime += end - start;
    TBPExecTime = totalExecTime - TBPExecTime;
    GBPExecTime = totalExecTime;
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
    
    // 3. GBP
    // 4. Transfer results from DPU
    // printf("Graph building phase:\n");
    pqueue_elem_t_mram *neighbors = malloc(pointAmt * neighborAmt * sizeof(pqueue_elem_t_mram));
    for (ADDRTYPE GBPbatch = 0; GBPbatch < leafIdSize; GBPbatch += nr_all_dpus) {
        ADDRTYPE max_dpus = min(leafIdSize - GBPbatch, nr_all_dpus);
        DPU_ASSERT(dpu_load_from_incbin(dpu_set, &dpu_binary_GBP, NULL));
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        hostExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        // Send data to DPUs. Note that redundant DPUs would be ignored
        DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(dimAmt), 0, &dimAmt, sizeof(uint32_t), DPU_XFER_ASYNC));
        loadLeavesIntoDPUsContext loadLeavesIntoDPUsContext_ctx = { .max_dpus = max_dpus, .points = points, .dpu_offset = dpu_offset, .tree = tree, .leafIds = leafIds + GBPbatch, .dimAmt = dimAmt };
        DPU_ASSERT(dpu_broadcast_to(dpu_set, STR(neighborAmt), 0, &neighborAmt, sizeof(uint32_t), DPU_XFER_ASYNC));
        DPU_ASSERT(dpu_callback(dpu_set, loadLeavesIntoDPUs, &loadLeavesIntoDPUsContext_ctx, DPU_CALLBACK_DEFAULT));
        // Execute on DPUs
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dataTransferhost2DPUTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));  // Can be replaced by `DPU_ASYNCHRONOUS`. Use `DPU_SYNCHRONOUS` here for measurement of actual DPU executing time. But it seems that the performance with both signs is similar
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dpuExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        // Get responses and update tree and treeSize
#ifdef PERF_EVAL_SIM
        perfcounter_t perfs[nr_all_dpus];
        uint32_t freqs[nr_all_dpus];
        getResponseFromGraphsContext getResponseFromGraphsContext_ctx = { .max_dpus = max_dpus, .neighbors = neighbors, .dpu_offset = dpu_offset, .tree = tree, .leafIds = leafIds + GBPbatch, .neighborAmt = neighborAmt, .perfs = perfs, .freqs = freqs };
#else
        getResponseFromGraphsContext getResponseFromGraphsContext_ctx = { .max_dpus = max_dpus, .neighbors = neighbors, .dpu_offset = dpu_offset, .tree = tree, .leafIds = leafIds + GBPbatch, .neighborAmt = neighborAmt };
#endif
        DPU_ASSERT(dpu_callback(dpu_set, getResponseFromGraphs, &getResponseFromGraphsContext_ctx, DPU_CALLBACK_DEFAULT));
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dataTransferDPU2hostTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
    }
    DPU_ASSERT(dpu_sync(dpu_set));
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time until graph building phase completed: %.6lfs\n", (end - start) / 1e6);
#endif

#ifdef PERF_EVAL
    gettimeofday(&timecheck, NULL);
#endif
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
    printf("[Host]  Total energy for k-graph construction: %.6lfJ\n", totalExecEnergy * ESU);
#endif
#ifdef PERF_EVAL
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    totalExecTime += end - start;
    hostExecTime += end - start;
    GBPExecTime = totalExecTime - GBPExecTime;
    printf("[Host]  Total time for k-graph construction: %.6lfs\n", totalExecTime / 1e6);
    printf("[Host]  Time for DPU execution: %.6lfs, host execution: %.6lfs, data transfer host2DPU: %.6lfs, data transfer DPU2host: %.6lfs\n", dpuExecTime / 1e6, hostExecTime / 1e6, dataTransferhost2DPUTime / 1e6, dataTransferDPU2hostTime / 1e6);
    printf("[Host]  Time for Tree Building Phase: %.6lfs, Graph Building Phase: %.6lfs\n", TBPExecTime / 1e6, GBPExecTime / 1e6);
#endif
    free(leafIds);

    // 5. Save results
    // printf("Result saving:\n");
    saveDataToFile(treeFileName, tree, sizeof(treeNode_t), treeIdSize);
    saveDataToFile(leafFileName, points, sizeof(ELEMTYPE), pointAmt * dimAmt);
    saveDataToFile(knnFileName, neighbors, sizeof(pqueue_elem_t_mram), pointAmt * neighborAmt);

    free(neighbors);
    free(tree);
    free(points);
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time for k-graph construction: %.3lfs\n", (end - start) / 1e6);
#endif
}


int main(int argc, char **argv) {
    struct dpu_set_t dpu_set;
    uint32_t nr_ranks;

    uint32_t dimAmt = 128;
    uint32_t neighborAmt = 10;
    uint32_t leafCapacity = 1000;
    uint32_t nb_mram = DPU_ALLOCATE_ALL;
    char *pointsFileName = "points.bin";
    char *treeFileName = "tree.bin";
    char *leafFileName = "leaf.bin";
    char *knnFileName = "knn.bin";
#ifdef PERF_EVAL
    uint64_t frequency = 450 << 20;
    parse_args(argc, argv, &dimAmt, &neighborAmt, &leafCapacity, &nb_mram, &frequency, &pointsFileName, &treeFileName, &leafFileName, &knnFileName);
#else
    parse_args(argc, argv, &dimAmt, &neighborAmt, &leafCapacity, &nb_mram, &pointsFileName, &treeFileName, &leafFileName, &knnFileName);
#endif

    printf("Allocating DPUs\n");
    DPU_ASSERT(dpu_alloc(nb_mram, "nrJobPerRank=64,dispatchOnAllRanks=true,cycleAccurate=true", &dpu_set));
    printf("DPUs allocated\n");
    DPU_ASSERT(dpu_get_nr_ranks(dpu_set, &nr_ranks));
    printf("Using %u MRAMs already loaded\n", nb_mram);

#ifdef PERF_EVAL
    allocated_and_compute(dpu_set, nr_ranks, dimAmt, neighborAmt, leafCapacity, pointsFileName, treeFileName, leafFileName, knnFileName);
#else
    allocated_and_compute(dpu_set, nr_ranks, dimAmt, neighborAmt, leafCapacity, pointsFileName, treeFileName, leafFileName, knnFileName);
#endif

    DPU_ASSERT(dpu_free(dpu_set));

    return 0;
}
