#include "OpenCLUtils.h"

#include "Cl/cl.h"

#include <vector>
#include <string>

cl_device_id device = nullptr;
cl_context context = nullptr;
cl_program program = nullptr;
cl_kernel kernel = nullptr;
cl_command_queue queue = nullptr;
cl_int err = -1;

bool VectorAdd(size_t local_size,
               std::vector<float>& vectorA,
               std::vector<float>& vectorB,
               std::vector<float>& output)
{
    if (vectorA.size() != vectorB.size())
    {
        printf("Invalid Input Sizes!");
        return false;
    }

    const size_t numValues = vectorA.size();
    const size_t dataSize = numValues * sizeof(float);

	cl_mem inputA = OpenCLUtils::create_input_buffer(context, vectorA.data(), dataSize);
	cl_mem inputB = OpenCLUtils::create_input_buffer(context, vectorB.data(), dataSize);
	cl_mem output_buffer = OpenCLUtils::create_output_buffer(context, dataSize);

    /* Create kernel arguments */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA); // <=====INPUT
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputB); // <=====INPUT
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer); // <=====OUTPUT
    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        return false;
    }

    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 1,
                                 NULL,
                                 &numValues,
                                 &local_size,
                                 0,
                                 NULL,
                                 NULL);

    if (err < 0)
    {
        perror("Couldn't enqueue the kernel");
        return false;
    }

    /* Read the kernel's output    */
    err = clEnqueueReadBuffer(queue,
                              output_buffer,
                              CL_TRUE,
                              0,
                              numValues * sizeof(float),
                              output.data(),
                              0,
                              NULL,
                              NULL);
    if (err < 0)
    {
        perror("Couldn't read the buffer");
        return false;
    }

	clReleaseKernel(kernel);
	clReleaseMemObject(inputA);
	clReleaseMemObject(inputB);
	clReleaseMemObject(output_buffer);
    return true;
}

bool InitializeDeviceAndContext()
{
	device = OpenCLUtils::create_device();
	if (!device)
	{
		return false;
	}

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		return false;
	}
    return true;
}

bool InitializeProgram()
{
	/* Build program */
	program = OpenCLUtils::build_program(context, device, "shaders/add_vectors.cl");
	if (!program)
		return false;

	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0)
	{
		perror("Couldn't create a command queue");
		return false;
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, "vectors_add", &err);
	if (err < 0)
	{
		perror("Couldn't create a kernel");
		return false;
	};
    return true;
}

int main() 
{
    const size_t numValues = 512;
    const size_t local_size = 4;

    if (!InitializeDeviceAndContext())
        return -1;

    if (!InitializeProgram())
        return -1;

    std::vector<float> vectorA(numValues, 1.0f);
    std::vector<float> vectorB(numValues, 2.0f);

    std::vector<float> output(numValues, 0.0f);

    if (!VectorAdd(local_size, vectorA, vectorB, output))
        return -1;

    /// Check Results ---------------------------------------------------------

    // Perform manual addition
    std::vector<float> checkOutput(numValues, 0.0f);
    for (size_t i = 0; i < numValues; ++i)
        checkOutput[i] = vectorA[i] + vectorB[i];

    bool success = true;
    for (size_t i = 0; i < numValues; ++i)
    {
        if (checkOutput[i] != output[i])
        {
            std::string msg = (std::to_string(checkOutput[i]) + " != " + std::to_string(output[i]));
			printf(msg.c_str());
			success = false;
            break;
        }
    }

    if (success)
    {
        printf("Correctly Added Vectors!");
    }
    else
    {
        printf("Failed to Correctly Add Vectors!");
    }
    /// -----------------------------------------------------------------------
    
    ///* Deallocate resources */
    
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
	return 0;
}