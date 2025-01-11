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

bool MatrixMult(size_t local_size,
                std::vector<float>& matrixA,
                std::vector<float>& matrixB,
                size_t M,
                size_t N,
                size_t K,
                std::vector<float>& output)
{
	cl_mem inputA = OpenCLUtils::create_input_buffer(context, matrixA.data(), matrixA.size() * sizeof(float));
	cl_mem inputB = OpenCLUtils::create_input_buffer(context, matrixB.data(), matrixB.size() * sizeof(float));
	cl_mem output_buffer = OpenCLUtils::create_output_buffer(context, output.size() * sizeof(float));

    /* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputB);
	err |= clSetKernelArg(kernel, 2, sizeof(int), &M);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &K);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_buffer);
    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        return false;
    }

	size_t global[2] = { M, K };
	size_t local[2] = { M, K };

    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 2,
                                 NULL,
                                 (const size_t*)&global,
                                 (const size_t*)&local,
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
                              output.size() * sizeof(float),
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
	program = OpenCLUtils::build_program(context, device, "shaders/matrix_mul.cl");
	if (!program)
		return false;

	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0)
	{
		perror("Couldn't create a command queue");
		return false;
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, "matrix_mul", &err);
	if (err < 0)
	{
		perror("Couldn't create a kernel");
		return false;
	};
    return true;
}

int main() 
{
	if (!InitializeDeviceAndContext())
        return -1;

    if (!InitializeProgram())
        return -1;

	std::vector<float> matrixA =
	{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12
	};
	std::vector<float> matrixB =
	{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15
	};

    const size_t rowsA = 4;
    const size_t columnsA = 3;
    const size_t rowsB = 3;
    const size_t columnB = 5;

    if (columnsA != rowsB)
        return -1;

	// Matrix dimensions: A[M x N] * B[N x K] = C[M x K]
	const size_t M = rowsA;     // Rows in A
	const size_t N = columnsA;  // Columns in A and Rows in B
	const size_t K = columnB;   // Columns in B

	// Result matrix
	std::vector<float> output(M * K, 0);

	const size_t local_size = 4;

    if (!MatrixMult(local_size, matrixA, matrixB, M, N, K, output))
        return -1;

    /// Check Results ---------------------------------------------------------

    // Perform Manual Multiplication
	std::vector<float> checkOutput(M * K, 0.0f);
	for (size_t i = 0; i < rowsA; ++i)
    {
		for (size_t j = 0; j < columnB; ++j)
        {
			for (size_t k = 0; k < columnsA; ++k)
            {
                size_t outputIndex = i * columnB + j;
                size_t indexA = i * columnsA + k;
                size_t indexB = k * columnB + j;
                checkOutput[outputIndex] += matrixA[indexA] * matrixB[indexB];
			}
		}
	}

	bool success = true;
	for (size_t i = 0; i < output.size(); ++i)
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
		printf("Correctly Multiplied the Matrices!");
	}
	else
	{
		printf("Failed to Multiply the Matrices!");
	}

    /// -----------------------------------------------------------------------
    
    ///* Deallocate resources */
    
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
	return 0;
}