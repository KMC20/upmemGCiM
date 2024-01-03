/*
Author: KMC20
Date: 2023/12/26
Function: Operations for evaluation of energy of UPMEM.
*/

#include "measureEnergy.h"

uint64_t rdmsr(int cpu, uint32_t reg) {  // Reference: https://github.com/lixiaobai09/intel_power_consumption_get/blob/master/powerget.c. Thanks!
    char buf[1024];
    sprintf(buf, "/dev/cpu/%d/msr", cpu);
    int msr_file = open(buf, O_RDONLY);
    if (msr_file < 0) {
        perror("rdmsr: open");
        return msr_file;
    }
    uint64_t data;
    if (pread(msr_file, &data, sizeof(data), reg) != sizeof(data)) {
        fprintf(stderr, "read msr register 0x%x error.\n", reg);
        perror("rdmsr: read msr");
        return -1;
    }
    close(msr_file);
    return data;
}

double getEnergyUnit() {
    uint64_t data = rdmsr(0, 0x606);
    return (double)1 / (1 << (data >> 8 & 0x1f));
}

uint32_t getEnergy(const uint32_t cpuId) {
    return rdmsr(cpuId, 0x619) & MSR_ENERGY_MASK;
}

uint32_t getNRSockets() {
    uint32_t socketAmount = 0;
    FILE *CPUinfo = NULL;
    if ((CPUinfo = popen("cat /proc/cpuinfo | grep 'physical id' | sort | uniq | wc -l", "r")) != NULL) {  // Get the amount of CPU sockets
        char buf[1024];
        if (fgets(buf, sizeof(buf), CPUinfo) == NULL)
            printf("fgets in 'getNRSockets' error\n");
        socketAmount = atoi(buf);
        pclose(CPUinfo);
        CPUinfo = NULL;
    } else {
        printf("popen cpu info error\n");
    }
    return socketAmount;
}

uint32_t getNRPhyCPUs() {
    uint32_t socketAmount = 0;
    FILE *CPUinfo = NULL;
    if ((CPUinfo = popen("cat /proc/cpuinfo | grep 'siblings' | uniq", "r")) != NULL) {  // Get the amount of CPU sockets
        char buf[1024];
        if (fgets(buf, sizeof(buf), CPUinfo) == NULL)
            printf("fgets in 'getNRPhyCPUs' error\n");
        socketAmount = atoi(strrchr(buf, ' '));
        pclose(CPUinfo);
        CPUinfo = NULL;
    } else {
        printf("popen cpu info error\n");
    }
    return socketAmount;
}