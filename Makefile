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

.PHONY : build_kernel_holder opencl_test clean_build_kernel_holder clean_opencl_test clean_all

opencl_test: opencl_test.c $(COMMON_DIR)/*.c
	make build_kernel_holder
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@ libshader_holder.a

build_kernel_holder: src/*.rs src/shader/*.cl
	cargo build
	mv ./target/debug/libshader_holder.a ./libshader_holder.a

clean_build_kernel_holder:
	rm -f libshader_holder.a shader_holder.h

clean_opencl_test:
	rm -f opencl_test

clean_all: clean_build_kernel_holder clean_opencl_test
