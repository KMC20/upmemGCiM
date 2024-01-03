/*
Author: KMC20
Date: 2023/12/1
Function: Operations for the graph building phase of GCiM.
*/

#ifndef GCIM_GRAPH_H
#define GCIM_GRAPH_H

#include "pqueue.h"  // libqueue: max heap
#include <string.h>  // memcpy, memmove
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>
#include <stdint.h>
#include <mram_unaligned.h>
#include <seqread.h>
#include "request.h"

#define increAddr(addr) addr + sizeof(pqueue_elem_t_mram)  // Increase the address by sizeof(ELEMTYPE *)

void graphBuilding(const __mram_ptr ELEMTYPE *const points, const ADDRTYPE pointAmt, const unsigned short dimAmt, const uint32_t neighborAmt, const ADDRTYPE pointNeighborStartAddr, __mram_ptr pqueue_elem_t_mram *neighbors);

#endif