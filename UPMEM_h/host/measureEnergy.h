/*
Author: KMC20
Date: 2023/12/26
Function: Operations for evaluation of energy of UPMEM.
*/

#ifndef UPMEM_MEASURE_ENERGY_H
#define UPMEM_MEASURE_ENERGY_H

#define _XOPEN_SOURCE 700  // To use popen, pread and pwrite with c11
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <stdint.h>
#include <string.h>

#define MSR_ENERGY_MASK 0xffffffff

double getEnergyUnit();
uint32_t getEnergy(const uint32_t cpuId);
uint32_t getNRSockets();
uint32_t getNRPhyCPUs();

#endif