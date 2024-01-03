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
#include "tree.h"
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
    ADDRTYPE treeIdSize = 0;
    // 1. Transfer data to DPU
    // 2. TBP (here, I record the left most address of points in each corresponding leaf node to reduce memory usage, which might be changed into `leafId * leafCapacity` for the future incremental updating)
    // printf("Tree building phase:\n");
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time for data preparation: %.3lfs\n", (end - start) / 1e6);
#endif
    ADDRTYPE *leafIds = malloc(MAX_TREE_SIZE * sizeof(ADDRTYPE));
    ADDRTYPE leafIdSize = 0;
    treeConstrDPU(tree, &treeIdSize, points, 0, pointAmt, dimAmt, leafCapacity, leafIds, &leafIdSize);
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time until tree building phase completed: %.3lfs\n", (end - start) / 1e6);
#endif

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
        // Send data to DPUs
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
    printf("[Host]  Total time until graph building phase completed: %.3lfs\n", (end - start) / 1e6);
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
