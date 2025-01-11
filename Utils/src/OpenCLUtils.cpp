#include "OpenCLUtils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

cl_device_id OpenCLUtils::create_device()
{
	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) 
    {
		perror("Couldn't identify a platform");
		return nullptr;
	}

	// Access a device
	// GPU
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) 
    {
		// CPU
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if (err < 0) 
    {
		perror("Couldn't access any devices");
		return nullptr;
	}
	return dev;
}

cl_program OpenCLUtils::build_program(cl_context ctx, cl_device_id dev, const char* filename)
{
    cl_program program;
    FILE* program_handle;
    char* program_buffer, * program_log;
    size_t program_size, log_size;
    int err;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL) 
    {
        perror("Couldn't find the program file");
		return nullptr;
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file

    Creates a program from the source code in the add_numbers.cl file.
    Specifically, the code reads the file's content into a char array
    called program_buffer, and then calls clCreateProgramWithSource.
    */
    program = clCreateProgramWithSource(ctx, 1,
        (const char**)&program_buffer, &program_size, &err);
    if (err < 0) 
    {
        perror("Couldn't create the program");
		return nullptr;
    }
    free(program_buffer);

    /* Build program

    The fourth parameter accepts options that configure the compilation.
    These are similar to the flags used by gcc. For example, you can
    define a macro with the option -DMACRO=VALUE and turn off optimization
    with -cl-opt-disable.
    */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) 
    {
        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
		return nullptr;
    }

    return program;
}

cl_mem OpenCLUtils::create_input_buffer(cl_context context, void* dataPtr, size_t dataSize)
{
    cl_int err = -1;
	cl_mem buffer = clCreateBuffer(context,
			                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			                        dataSize,
			                        dataPtr,
			                        &err);

    if (err < 0)
    {
        return nullptr;
    }
    return buffer;
}

cl_mem OpenCLUtils::create_output_buffer(cl_context context, size_t dataSize)
{
	cl_int err = -1;
	cl_mem buffer = clCreateBuffer(context,
                                   CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                   dataSize, 
                                   NULL, 
                                   &err);

	if (err < 0)
	{
		return nullptr;
	}
	return buffer;
}