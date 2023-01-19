CC = gcc
CCFLAGS=-O3 -lm
LIBS= -lOpenCL

COMMON_DIR=./utils

DEVICE = CL_DEVICE_TYPE_DEFAULT
CCFLAGS += -D DEVICE=$(DEVICE)

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

opencl_test: opencl_test.c $(COMMON_DIR)/*.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@

clean:
	rm -f opencl_test
