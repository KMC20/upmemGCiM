# UPMEM Implementations for GCiM (Graph Construction in Memory)

This repo provides two implementations for GCiM (Graph Construction in Memory) for nearest neighbor search in high dimension space. The version in UPMEM_d constructs the tree structure on the host side while the other in UPMEM_h constructs the tree structure on DPU side. Both construct subgraphs on DPU side.

## How to test

The following commands will run small example tests:

```
bash run.sh
```

The example dataset is quantized from SIFT10K to 16bits. You can change the base type with `ELEMTYPE` in common/inc/request.h in UPMEM_d or UPMEM_h for your own requirements.

## How to use on a new dataset

Change the parameter `-p` in the Makefile in UPMEM_d or UPMEM_h. The default one is `datasets/exampleData`.

## How to check the results

Find the result tree, leaves and k-graph files in the directory `ckpts` in UPMEM_d or UPMEM_h if you run the example `run.sh`

## Thanks

 * Library of priority queue. The implementations of priority queue in dpu/libpqueue in UPMEM_d and UPMEM_h are modified to suit the demands of UPMEM from the implementaion here: https://github.com/vy/libpqueue
 * Pseudo random generator. The implementation of `randGen` in UPMEM_d/dpu/src/tree.c for pseudo random generation is modified from the implementaion here: https://github.com/0/msp430-rng
 * MSR fetching. The implementations of `rdmsr` and related functions in host/measureEnergy.c in UPMEM_d and UPMEM_h for energy measurement are modified from the implementaion here: https://github.com/lixiaobai09/intel_power_consumption_get/blob/master/powerget.c
 * The basic directory structure of the repo is learned from the UPIS project: https://github.com/upmem/usecase_UPIS

## Reference

If you feel this repo is useful for you, not hesitate to star this!😀 And it is really kind of you to cite this repo in your paper or project.

If you feel GCiM is interesting, please cite:

```
@INPROCEEDINGS{9586221,  author={He, Lei and Liu, Cheng and Wang, Ying and Liang, Shengwen and Li, Huawei and Li, Xiaowei},  booktitle={2021 58th ACM/IEEE Design Automation Conference (DAC)},   title={GCiM: A Near-Data Processing Accelerator for Graph Construction},   year={2021},  volume={},  number={},  pages={205-210},  doi={10.1109/DAC18074.2021.9586221}}
```
