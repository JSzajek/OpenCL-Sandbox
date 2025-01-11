#include "OpenCLUtils.h"
#include "OpenCVUtils.h"

#include "opencv2/opencv.hpp"

#include "Cl/cl.h"

#include <vector>
#include <string>

cl_device_id device = nullptr;
cl_context context = nullptr;
cl_program program = nullptr;
cl_kernel kernel = nullptr;
cl_command_queue queue = nullptr;
cl_int err = -1;

enum class KernelType : uint8_t
{
	KernelType_Sobel = 0,
	KernelType_Perwitt,
	KernelType_Scharr,
	KernelType_Laplacian,
	KernelType_Gaussian,
	KernelType_Sharpen,
	KernelType_EdgeEnhancement,
	KernelType_BoxBlur
};

void GetKernel(KernelType type, 
			   std::vector<float>& kernelX,
			   std::vector<float>& kernelY);

bool Cartoonize(const cv::Mat& input,
				std::vector<float>& kernel_x,
				std::vector<float>& kernel_y,
				int quantization_lvls,
				float edge_threshold,
                cv::Mat& output)
{
	const size_t inputDataSize = input.cols * input.rows * input.channels() * sizeof(unsigned char);
	const size_t kernelDataSize = kernel_x.size() * sizeof(float);
	const size_t outputDataSize = input.cols * input.rows * output.channels() * sizeof(unsigned char);
	cl_mem inputA = OpenCLUtils::create_input_buffer(context, input.data, inputDataSize);
	cl_mem kernelX = OpenCLUtils::create_input_buffer(context, kernel_x.data(), kernelDataSize);
	cl_mem kernelY = OpenCLUtils::create_input_buffer(context, kernel_y.data(), kernelDataSize);
	cl_mem output_buffer = OpenCLUtils::create_output_buffer(context, outputDataSize);

	const int width = input.cols;
	const int height = input.rows;
	const int channels = input.channels();

    /* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &kernelX);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &kernelY);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &quantization_lvls);
	err |= clSetKernelArg(kernel, 4, sizeof(float), &edge_threshold);
	err |= clSetKernelArg(kernel, 5, sizeof(int), &width);
	err |= clSetKernelArg(kernel, 6, sizeof(int), &height);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &output_buffer);
    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        return false;
    }

	size_t global[2] = { width, height };

    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 2,
                                 NULL,
                                 (const size_t*)&global,
                                 NULL,
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
                              outputDataSize,
                              output.data,
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
	program = OpenCLUtils::build_program(context, device, "shaders/cartoon_img.cl");
	if (!program)
		return false;

	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0)
	{
		perror("Couldn't create a command queue");
		return false;
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, "cartoonize", &err);
	if (err < 0)
	{
		perror("Couldn't create a kernel");
		return false;
	};
    return true;
}

int main()
{
	KernelType kernelType = KernelType::KernelType_Sobel;
	const int quantization_levels = 16;
	const float edge_threshold = 150.0f;

	if (!InitializeDeviceAndContext())
        return -1;

    if (!InitializeProgram())
        return -1;

	cv::Mat inputImg = cv::imread("content/test.jpg");
	if (!OpenCVUtils::ConvertType(inputImg, CV_8UC4))
	{
		printf("Failed to Convert Output Image Type!");
		return -1;
	}

	cv::Mat outputImg(inputImg.rows, inputImg.cols, inputImg.type(), cv::Scalar(0, 0, 0));

	std::vector<float> kernel_x;
	std::vector<float> kernel_y;
	GetKernel(kernelType, kernel_x, kernel_y);

	cv::Mat inputImgRGBA;
	cv::cvtColor(inputImg, inputImgRGBA, cv::COLOR_BGRA2RGBA);
    if (!Cartoonize(inputImgRGBA, kernel_x, kernel_y, quantization_levels, edge_threshold, outputImg))
        return -1;

    /// Check Results ---------------------------------------------------------

	const size_t combinedWidth = inputImg.cols + outputImg.cols;
	cv::Mat combinedImage(cv::Size(combinedWidth, inputImg.rows), inputImg.type());

	// Copy the first image into the left side of the combined image
    inputImg.copyTo(combinedImage(cv::Rect(0, 0, inputImg.cols, inputImg.rows)));

	// Copy the second image into the right side of the combined image
	cv::cvtColor(outputImg, outputImg, cv::COLOR_RGBA2BGRA);
	outputImg.copyTo(combinedImage(cv::Rect(inputImg.cols, 0, outputImg.cols, outputImg.rows)));
    
	cv::imshow("Side by Side Result", combinedImage);
	cv::waitKey();
    
    ///* Deallocate resources */
    
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
	return 0;
}


void GetKernel(KernelType type, 
			   std::vector<float>& kernelX,
			   std::vector<float>& kernelY)
{
	switch (type)
	{
		case KernelType::KernelType_Sobel:
		default:
		{
			kernelX =
			{
				-1, 0, 1,
				-2, 0, 2,
				-1, 0, 1
			};

			kernelY =
			{
				-1, -2, -1,
				 0,  0,  0,
				 1,  2,  1
			};
			break;
		}
		case KernelType::KernelType_Perwitt:
		{
			kernelX =
			{
				-1, 0, 1,
				-1, 0, 1,
				-1, 0, 1
			};

			kernelY =
			{
				-1, -1, -1,
				 0,  0,  0,
				 1,  1,  1
			};
			break;
		}
		case KernelType::KernelType_Scharr:
		{
			kernelX =
			{
				-3, 0,  3,
			   -10, 0, 10,
				-3, 0,  3
			};

			kernelY =
			{
				-3, -10, -3,
				 0,   0,  0,
				 3,  10,  3
			};
			break;
		}
		case KernelType::KernelType_Laplacian:
		{
			kernelX =
			{
				 0, -1,  0,
				-1,  4, -1,
				 0, -1,  0
			};

			kernelY =
			{
				-1, -1, -1,
				-1,  8, -1,
				-1, -1, -1
			};
			break;
		}
		case KernelType::KernelType_Gaussian:
		{
			kernelX =
			{
				 1, 2, 1,
				 2, 4, 2,
				 1, 2, 1
			};

			kernelY = kernelX;
			break;
		}
		case KernelType::KernelType_Sharpen:
		{
			kernelX =
			{
				 0, -1,  0,
				-1,  5, -1,
				 0, -1,  0
			};

			kernelY = kernelX;
			break;
		}
		case KernelType::KernelType_EdgeEnhancement:
		{
			kernelX =
			{
				-1, -1, -1,
				-1,  9, -1,
				-1, -1, -1
			};

			kernelY = kernelX;
			break;
		}
		case KernelType::KernelType_BoxBlur:
		{
			kernelX =
			{
				1, 1, 1,
				1, 1, 1,
				1, 1, 1
			};

			kernelY = kernelX;
			break;
		}
	}
}
