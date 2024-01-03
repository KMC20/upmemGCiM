/*
Author: KMC20
Date: 2023/12/16
Function: Operations for the tree building phase on host of GCiM.
*/

#ifndef GCIM_TREE_H
#define GCIM_TREE_H

#include <stdint.h>
#include "request.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

/***************************************************************************************************************************************************************************************************************/
/******************************************************************************** This part is copied from `stdbool.h` of upmem ********************************************************************************/
/***************************************************************************************************************************************************************************************************************/
/**
 * @def true
 * @brief The <code>true</code> constant, represented by <code>1</code>
 */
#define true 1
/**
 * @def false
 * @brief The <code>false</code> constant, represented by <code>0</code>
 */
#define false 0
/***************************************************************************************************************************************************************************************************************/

MEAN_VALUE_TYPE accumulatorIndependent(const ELEMTYPE *const points, const ADDRTYPE left, const ADDRTYPE right, const uint32_t dim, const uint32_t dimAmt);
ADDRTYPE meanSpliterIndependent(ELEMTYPE *points, const uint32_t pointSize, const ADDRTYPE left, const ADDRTYPE right, const MEAN_VALUE_TYPE mean, const uint32_t dim, const uint32_t dimAmt);
void treeConstrDPU(treeNode_t *tree, ADDRTYPE *treeSizeRes, ELEMTYPE *points, const ADDRTYPE treeBaseAddr, const uint32_t pointAmt, const unsigned short dimAmt, const unsigned short leafCapacity, ADDRTYPE *leafIds, ADDRTYPE *leafIdSizeRes);

#endif