/*
Author: KMC20
Date: 2023/12
Function: Public definitions for GCiM.
*/

#ifndef REQUEST_H
#define REQUEST_H

#include <stdint.h>

/* Public definitions for GCiM are listed below */
typedef unsigned short ELEMTYPE;
typedef uint32_t ADDRTYPE;  // sizeof(uint32_t) == sizeof(ELEMTYPE *)
#define MRAM_SIZE (62 << 20)
#define MRAM_ALIGN_BYTES 8
#define ADDRTYPE_NULL 0  // NULL pointer for ADDRTYPE
// Used for tree
typedef uint32_t MEAN_VALUE_TYPE;
typedef struct treeNodeType {
    ADDRTYPE left;
    ADDRTYPE right;
    MEAN_VALUE_TYPE mean;  // For leaf nodes, this domain is used as the left most addr on all points
    unsigned short dim;  // For leaf nodes, this domain is used as the point size of this leaf
} treeNode_t;
// Used for graph
// Used for priority queue
typedef unsigned long long pqueue_pri_t;
typedef struct {
    pqueue_pri_t pri;
	ADDRTYPE val;
	// uint32_t pos;
} pqueue_elem_t_mram;

#endif // REQUEST_H
