#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"
#define ASSERTF_DEF_ONCE
#include "assertf.h"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern int output_device_info(cl_device_id);
char *getKernelSource(char *);

int main(int argc, char **argv)
{
    int err; // error code returned from OpenCL calls

    // Find number of platforms
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return 1;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    cl_device_id device_id;
    for (int i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }
    if (device_id == NULL)
        checkError(err, "Finding a device");

    err = output_device_info(device_id);
    checkError(err, "Printing device output");

    // Create a compute context
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    char *kernel_source = getKernelSource("./kernel.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    unsigned int width = 4;
    unsigned int height = 4;
    unsigned int count = 3;

    unsigned int buffer_input_size = sizeof(unsigned int) * width * height * count;
    unsigned int *buffer_input = (unsigned int *)malloc(buffer_input_size);

    // ? "random values"
    for (unsigned int i = 0; i < width * height; i += 1)
    {
        for (uint l = 0; l < count; l += 1)
        {
            buffer_input[i + width * height * l] = 0x00FF0000 + ((l + 0xAA) % 0xFF) + ((i % 0xFF) << 24);
        }
    }

    printf("(random) input buffer:\n");
    for (uint i = 0; i < width * height * count; i += 1)
    {
        printf("0x%08x", buffer_input[i]);
        if ((i + 1) % (width * height) == 0)
        {
            printf("\n");
        }
        else
        {
            printf(", ");
        }
    }

    unsigned int buffer_output_size = sizeof(unsigned int) * width * height;
    unsigned int *buffer_output = (unsigned int *)malloc(buffer_output_size);

    cl_kernel ko_combine_layers = clCreateKernel(program, "combine_layers", &err);
    checkError(err, "Creating kernel for 'combine_layers'");

    cl_mem d_buffer_input = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_input_size, NULL, &err);
    checkError(err, "Creating buffer 'd_buffer_input'");

    cl_mem d_buffer_ouput = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_output_size, NULL, &err);
    checkError(err, "Creating buffer 'd_buffer_ouput'");

    err = clEnqueueWriteBuffer(commands, d_buffer_input, CL_TRUE, 0, buffer_input_size, buffer_input, 0, NULL, NULL);
    checkError(err, "Copying buffer_input to device at d_buffer_input");

    err = clSetKernelArg(ko_combine_layers, 0, sizeof(cl_mem), &d_buffer_input);
    err |= clSetKernelArg(ko_combine_layers, 1, sizeof(cl_mem), &d_buffer_ouput);
    err |= clSetKernelArg(ko_combine_layers, 2, sizeof(unsigned int), &width);
    err |= clSetKernelArg(ko_combine_layers, 3, sizeof(unsigned int), &height);
    err |= clSetKernelArg(ko_combine_layers, 4, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");

    const size_t global[2] = {width, height};
    err = clEnqueueNDRangeKernel(
        commands,
        ko_combine_layers,
        2,
        NULL,
        global,
        NULL,
        0,
        NULL,
        NULL);
    checkError(err, "Enqueueing kernel");

    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    err = clEnqueueReadBuffer(commands, d_buffer_ouput, CL_TRUE, 0, buffer_output_size, buffer_output, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    printf("output buffer:\n");
    for (uint i = 0; i < width * height; i += 1)
    {
        printf("0x%08x, ", buffer_output[i]);
    }
    printf("\n");

    assertf(buffer_output[0] == 0x00f600ff, "0x%08x", buffer_output[0]);
    assertf(buffer_output[1] == 0x00f600ff, "0x%08x", buffer_output[1]);
    assertf(buffer_output[2] == 0x01f600ff, "0x%08x", buffer_output[2]);
    assertf(buffer_output[3] == 0x02f600ff, "0x%08x", buffer_output[3]);
    assertf(buffer_output[4] == 0x03f600ff, "0x%08x", buffer_output[4]);
    assertf(buffer_output[5] == 0x04f600ff, "0x%08x", buffer_output[5]);
    assertf(buffer_output[6] == 0x05f600ff, "0x%08x", buffer_output[6]);
    assertf(buffer_output[7] == 0x06f600ff, "0x%08x", buffer_output[7]);
    assertf(buffer_output[8] == 0x07f600ff, "0x%08x", buffer_output[8]);
    assertf(buffer_output[9] == 0x08f600ff, "0x%08x", buffer_output[9]);
    assertf(buffer_output[10] == 0x09f600ff, "0x%08x", buffer_output[10]);
    assertf(buffer_output[11] == 0x0af600ff, "0x%08x", buffer_output[11]);
    assertf(buffer_output[12] == 0x0bf600ff, "0x%08x", buffer_output[12]);
    assertf(buffer_output[13] == 0x0cf600ff, "0x%08x", buffer_output[13]);
    assertf(buffer_output[14] == 0x0df600ff, "0x%08x", buffer_output[14]);
    assertf(buffer_output[15] == 0x0ef600ff, "0x%08x", buffer_output[15]);

    // cleanup and exit
    clReleaseMemObject(d_buffer_input);
    clReleaseMemObject(d_buffer_ouput);
    clReleaseKernel(ko_combine_layers);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(buffer_input);
    free(buffer_output);

    return 0;
}

char *getKernelSource(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Could not open kernel source file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) + 1;
    rewind(file);

    char *source = (char *)malloc(sizeof(char) * len);
    if (!source)
    {
        fprintf(stderr, "Error: Could not allocate memory for source string\n");
        fclose(file);
        exit(1);
    }

    fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
}
