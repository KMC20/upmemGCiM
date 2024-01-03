BUILDDIR ?= build

HOST_BINARY=${BUILDDIR}/host_app
HOST_SOURCES=$(wildcard host/*.c)
HOST_HEADERS=$(wildcard host/*.h)

DPU_SOURCES=$(wildcard dpu/src/*.c dpu/libpqueue/src/pqueue.c)
DPU_HEADERS=$(wildcard dpu/inc/*.h dpu/libpqueue/src/pqueue.h)
DPU_MAIN_TBP_ACCUMULATOR=dpu/TBP_accumulator.c
DPU_MAIN_TBP_MEANSPLITER=dpu/TBP_meanSpliter.c
DPU_MAIN_TBP=dpu/TBP.c
DPU_MAIN_GBP=dpu/GBP.c
DPU_BINARY_TBP_ACCUMULATOR=${BUILDDIR}/dpu_task_TBP_accumulator
DPU_BINARY_TBP_MEANSPLITER=${BUILDDIR}/dpu_task_TBP_meanSpliter
DPU_BINARY_TBP=${BUILDDIR}/dpu_task_TBP
DPU_BINARY_GBP=${BUILDDIR}/dpu_task_GBP

COMMONS_HEADERS=$(wildcard common/inc/*.h)

OUTPUT_FILE=${BUILDDIR}/output.txt
PLOTDATA_FILE=${BUILDDIR}/plotdata.csv

CHECK_FORMAT_FILES=${HOST_SOURCES} ${HOST_HEADERS} ${DPU_SOURCES} ${DPU_HEADERS} ${COMMONS_HEADERS} ${DPU_MAIN_TBP_ACCUMULATOR} ${DPU_MAIN_TBP_MEANSPLITER} ${DPU_MAIN_TBP} ${DPU_MAIN_GBP}
CHECK_FORMAT_DEPENDENCIES=$(addsuffix -check-format,${CHECK_FORMAT_FILES})

NR_TASKLETS ?= 24  # For GIST1M, set this no larger than 10! (Because the dimension is larger than SIFT*, and the WRAM buffer allocated for points is linear to the dimension)

__dirs := $(shell mkdir -p ${BUILDDIR})

.PHONY: all clean run plotdata check check-format

all: ${HOST_BINARY} ${DPU_BINARY_TBP_ACCUMULATOR} ${DPU_BINARY_TBP_MEANSPLITER} ${DPU_BINARY_TBP} ${DPU_BINARY_GBP}
clean:
	rm -rf ${BUILDDIR}

###
### HOST APPLICATION
###
CFLAGS=-g -Wall -Werror -Wextra -O3 -std=c11 `dpu-pkg-config --cflags dpu` -Ihost/inc -Icommon/inc -DNR_TASKLETS=${NR_TASKLETS}
LDFLAGS=`dpu-pkg-config --libs dpu` -fopenmp

${HOST_BINARY}: ${HOST_SOURCES} ${HOST_HEADERS} ${COMMONS_HEADERS} ${DPU_BINARY_TBP_ACCUMULATOR} ${DPU_BINARY_TBP_MEANSPLITER} ${DPU_BINARY_TBP} ${DPU_BINARY_GBP}
	$(CC) -o $@ ${HOST_SOURCES} $(LDFLAGS) $(CFLAGS) -DDPU_BINARY_TBP_ACCUMULATOR=\"$(realpath ${DPU_BINARY_TBP_ACCUMULATOR})\" \
													 -DDPU_BINARY_TBP_MEANSPLITER=\"$(realpath ${DPU_BINARY_TBP_MEANSPLITER})\" \
													 -DDPU_BINARY_TBP=\"$(realpath ${DPU_BINARY_TBP})\" \
													 -DDPU_BINARY_GBP=\"$(realpath ${DPU_BINARY_GBP})\" -DPERF_EVAL -DENERGY_EVAL
# 	$(CC) -o $@ ${HOST_SOURCES} $(LDFLAGS) $(CFLAGS) -DDPU_BINARY_TBP_ACCUMULATOR=\"$(realpath ${DPU_BINARY_TBP_ACCUMULATOR})\" \
# 													 -DDPU_BINARY_TBP_MEANSPLITER=\"$(realpath ${DPU_BINARY_TBP_MEANSPLITER})\" \
# 													 -DDPU_BINARY_TBP=\"$(realpath ${DPU_BINARY_TBP})\" \
# 													 -DDPU_BINARY_GBP=\"$(realpath ${DPU_BINARY_GBP})\"

###
### DPU BINARY
###
# DPU_FLAGS=-g -O2 -Wall -Werror -Wextra -flto=thin -Idpu/inc -Icommon/inc -Idpu/libpqueue/src -DNR_TASKLETS=${NR_TASKLETS} -DSTACK_SIZE_DEFAULT=256
DPU_FLAGS=-g -O2 -Wall -Werror -Wextra -flto=thin -Idpu/inc -Icommon/inc -Idpu/libpqueue/src -DNR_TASKLETS=${NR_TASKLETS} -DSTACK_SIZE_DEFAULT=256 -DPERF_EVAL

${DPU_BINARY_TBP_ACCUMULATOR}: ${DPU_MAIN_TBP_ACCUMULATOR} ${DPU_SOURCES} ${DPU_HEADERS} ${COMMONS_HEADERS}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} ${DPU_SOURCES} ${DPU_MAIN_TBP_ACCUMULATOR} -o $@

${DPU_BINARY_TBP_MEANSPLITER}: ${DPU_MAIN_TBP_MEANSPLITER} ${DPU_SOURCES} ${DPU_HEADERS} ${COMMONS_HEADERS}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} ${DPU_SOURCES} ${DPU_MAIN_TBP_MEANSPLITER} -o $@

${DPU_BINARY_TBP}: ${DPU_MAIN_TBP} ${DPU_SOURCES} ${DPU_HEADERS} ${COMMONS_HEADERS}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} ${DPU_SOURCES} ${DPU_MAIN_TBP} -o $@

${DPU_BINARY_GBP}: ${DPU_MAIN_GBP} ${DPU_SOURCES} ${DPU_HEADERS} ${COMMONS_HEADERS}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} ${DPU_SOURCES} ${DPU_MAIN_GBP} -o $@

###
### EXECUTION & TEST
###
run: all
	${HOST_BINARY} -p datasets/exampleData -t ckpts/exampleTree -l ckpts/exampleLeaves -k ckpts/exampleGraph -D 128 -K 10 -L 1000 >> ${OUTPUT_FILE} 2>&1

check:
	cat ${OUTPUT_FILE} | grep "matchs found" | diff output.txt -  # Useless

plotdata:
	echo "Mcc" > ${PLOTDATA_FILE}
	cat ${OUTPUT_FILE} | grep "average execution time" | sed 's/\[DPU\]  average execution time.*= .* ms (\(.*\) Mcc)/\1/' >> ${PLOTDATA_FILE}  # Useless

%-check-format: %
	clang-format $< | diff -y --suppress-common-lines $< -  # Useless

check-format: ${CHECK_FORMAT_DEPENDENCIES}  # Useless
