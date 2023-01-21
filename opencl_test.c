#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <assert.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

// pick up device type from compiler command line or from
// the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern double wtime(); // returns time since some fixed past point (wtime.c)
extern int output_device_info(cl_device_id);

char *getKernelSource(char *);

#define TOL (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024) // length of vectors a, b, and c

// #define VADD 1

int main(int argc, char **argv)
{
    int err; // error code returned from OpenCL calls

    float *h_a = (float *)malloc(LENGTH * sizeof(float)); // a vector
    float *h_b = (float *)malloc(LENGTH * sizeof(float)); // b vector
    float *h_c = (float *)malloc(LENGTH * sizeof(float)); // c vector (a+b) returned from the compute device

    unsigned int correct; // number of correct results

    size_t global; // global domain size

    cl_device_id device_id;    // compute device id
    cl_context context;        // compute context
    cl_command_queue commands; // compute command queue
    cl_program program;        // compute program
    cl_kernel ko_vadd;         // compute kernel

    cl_mem d_a; // device memory used for the input  a vector
    cl_mem d_b; // device memory used for the input  b vector
    cl_mem d_c; // device memory used for the output c vector

    // Fill vectors a and b with random float values
    int i = 0;
    int count = LENGTH;
    for (i = 0; i < count; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Set up platform and GPU device

    cl_uint numPlatforms;

    // Find number of platforms
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
    for (i = 0; i < numPlatforms; i++)
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
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    char *kernel_source = getKernelSource("./kernel.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
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

#ifdef VADD

    // Create the compute kernel from the program
    ko_vadd = clCreateKernel(program, "vadd", &err);
    checkError(err, "Creating kernel");

    // Create the input (a, b) and output (c) arrays in device memory
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_a");

    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_b");

    d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_c");

    // Write a and b vectors into compute device memory
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");

    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, h_b, 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    // Set the arguments to our compute kernel
    err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");

    double rtime = wtime();

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    global = count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    rtime = wtime() - rtime;
    printf("\nThe kernel ran in %lf seconds\n", rtime);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    // Test the results
    correct = 0;
    float tmp;

    for (i = 0; i < count; i++)
    {
        tmp = h_a[i] + h_b[i];     // assign element i of a+b to tmp
        tmp -= h_c[i];             // compute deviation of expected and output result
        if (tmp * tmp < TOL * TOL) // correct if square deviation is less than tolerance squared
            correct++;
        else
        {
            printf(" tmp %f h_a %f h_b %f h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
        }
    }

    // summarise results
    printf("C = A+B:  %d out of %d results were correct.\n", correct, count);

    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(ko_vadd);

    free(h_a);
    free(h_b);
    free(h_c);
#else
    unsigned int width = 256;
    unsigned int height = 256;
    unsigned int ccount = 3;

    unsigned buffer_input_size = sizeof(unsigned int) * width * height * ccount;
    unsigned int *buffer_input = (unsigned int *)malloc(buffer_input_size);

    for (unsigned int i = 0; i < width * height; i += 1)
    {
        buffer_input[i] = 0xFF0000FF;
        buffer_input[i + width * height] = 0x00FF00FF;
        buffer_input[i + width * height * 2] = 0x0F1F1AFF;
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
    err |= clSetKernelArg(ko_combine_layers, 4, sizeof(unsigned int), &ccount);
    checkError(err, "Setting kernel arguments");

    const size_t gglobal[2] = {width, height};
    err = clEnqueueNDRangeKernel(
        commands,
        ko_combine_layers,
        2,
        NULL,
        gglobal,
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

    assert(buffer_output[0] == 0x5A5F08FF);
    // for (uint i = 0; i < (width * height); i += 1)
    //{
    //     printf("(cpu)\t%d -> %08X\n", i, buffer_output[i]);
    // }

    // cleanup
    clReleaseMemObject(d_buffer_input);
    clReleaseMemObject(d_buffer_ouput);
    clReleaseKernel(ko_combine_layers);

    free(buffer_input);
    free(buffer_output);

#endif

    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

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
