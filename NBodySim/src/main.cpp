#include "OpenCLUtils.h"
#include "OpenCVUtils.h"
#include "RandomUtils.h"
#include "Timer.h"

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

struct Vector4f
{
public:
	Vector4f() = default;

	Vector4f(float _x, float _y, float _z, float _w)
		: x(_x),
		y(_y),
		z(_z),
		w(_w)
	{
	}

public:
	float x = 0;
	float y = 0;
	float z = 0;
	float w = 0;
};

struct Vector2f
{
public:
	Vector2f() = default;

	Vector2f(float _x, float _y)
		: x(_x),
		y(_y)
	{
	}
public:
	inline float Length() { return std::sqrt(x * x + y * y); }
public:
	float x = 0;
	float y = 0;
};

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
	program = OpenCLUtils::build_program(context, device, "shaders/nbody.cl");
	if (!program)
		return false;

	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0)
	{
		perror("Couldn't create a command queue");
		return false;
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, "simulate", &err);
	if (err < 0)
	{
		perror("Couldn't create a kernel");
		return false;
	};
    return true;
}

int main()
{
	constexpr size_t Num_Particles = 50;

	const cv::Scalar BackgroundColor(57, 36, 36);
	const cv::Scalar BodyColor(204, 232, 169);
	const cv::Scalar LineColor(205, 44, 44);
	const int LineThickness = 2;

	const float MinRadius = 5;
	const float MaxRadius = 35;

	const float MinMass = 5;
	const float MaxMass = 100;

	const float GravitationConstant = 0.3;
	const float BounceFactor = 0.1f;

	const size_t MaxX = 512;
	const float HalfMaxX = MaxX * 0.5f;
	const size_t MaxY = 512;
	const float HalfMaxY = MaxY * 0.5f;

	Vector4f Bounds;
	Bounds.x = 0;			// MinX
	Bounds.y = MaxX;		// MaxX
	Bounds.z = 0;			// MinY
	Bounds.w = MaxY;		// MaxY

	std::vector<Vector2f> Positions(Num_Particles);
	std::vector<Vector2f> Velocities(Num_Particles);
	std::vector<Vector2f> Accelerations(Num_Particles);
	std::vector<float> Radii(Num_Particles);
	std::vector<float> Masses(Num_Particles);

	const auto GetRandFloat = [](size_t Max)
	{
		return static_cast<float>(rand() % MaxX);
	};

	// Initialize particles
	for (int i = 0; i < Num_Particles; i++)
	{
		Positions[i]	= Vector2f(RandUtils::RandomRange<size_t>(0, MaxX),
								   RandUtils::RandomRange<size_t>(0, MaxY));

		Radii[i] = RandUtils::RandomRange<float>(MinRadius, MaxRadius);
		Masses[i] = RandUtils::RandomRange<float>(MinMass, MaxMass);
	}

	if (!InitializeDeviceAndContext())
        return -1;

    if (!InitializeProgram())
        return -1;

	size_t float2BufferDataSize = Num_Particles * sizeof(Vector2f);
	cl_mem positionsBuffer = OpenCLUtils::create_input_buffer(context, Positions.data(), float2BufferDataSize);
	cl_mem velocitiesBuffer = OpenCLUtils::create_input_buffer(context, Velocities.data(), float2BufferDataSize);
	cl_mem accelerationsBuffer = OpenCLUtils::create_input_buffer(context, Accelerations.data(), float2BufferDataSize);

	size_t floatBufferDataSize = Num_Particles * sizeof(float);
	cl_mem radiiBuffer = OpenCLUtils::create_input_buffer(context, Radii.data(), floatBufferDataSize);
	cl_mem massesBuffer = OpenCLUtils::create_input_buffer(context, Masses.data(), floatBufferDataSize);

	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &positionsBuffer);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &velocitiesBuffer);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &accelerationsBuffer);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &radiiBuffer);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &massesBuffer);
	err |= clSetKernelArg(kernel, 6, sizeof(int), &Num_Particles);
	err |= clSetKernelArg(kernel, 7, sizeof(float), &GravitationConstant);
	err |= clSetKernelArg(kernel, 8, sizeof(float), &BounceFactor);
	err |= clSetKernelArg(kernel, 9, sizeof(Vector4f), &Bounds);
	if (err < 0)
	{
		perror("Couldn't create a kernel argument");
		return false;
	}

	size_t global = Num_Particles;

	const std::string winName = "NBody Simulation";
	cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
	cv::Mat outputImg(MaxY, MaxX, CV_8UC4, BackgroundColor);

	Timer gpuBufferReadTimer;
	Timer drawTimer;
	auto previousTime = std::chrono::high_resolution_clock::now();

	float deltaTime_s = 0.01f;
	while (true)
	{
		gpuBufferReadTimer.Start();

		// Update delta time --------------------------------------------------
		err = clSetKernelArg(kernel, 5, sizeof(float), &deltaTime_s);
		if (err < 0)
		{
			perror("Couldn't create a kernel argument");
			return false;
		}
		// --------------------------------------------------------------------

		err = clEnqueueNDRangeKernel(queue,
									 kernel,
									 1,
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
								  positionsBuffer,
								  CL_FALSE,
								  0,
								  float2BufferDataSize,
								  Positions.data(),
								  0,
								  NULL,
								  NULL);

		err |= clEnqueueReadBuffer(queue,
								   velocitiesBuffer,
								   CL_FALSE,
								   0,
								   float2BufferDataSize,
								   Velocities.data(),
								   0,
								   NULL,
								   NULL);
		if (err < 0)
		{
			perror("Couldn't read the buffer");
			return false;
		}

		clFinish(queue);

		const double gpuBufferTime_ms = gpuBufferReadTimer.Elapsed_ms();

		// Visualization logic
		drawTimer.Start();
		{
			outputImg.setTo(BackgroundColor);
			for (size_t x = 0; x < Num_Particles; ++x)
			{
				const Vector2f position = Positions[x];
				const float radius = Radii[x];
				cv::circle(outputImg, 
						   cv::Point(position.x, position.y),
						   static_cast<int>(radius), 
						   BodyColor,
						   LineThickness, 
						   cv::LineTypes::FILLED);

				Vector2f velocity = Velocities[x];
				const float speed = velocity.Length();
				cv::line(outputImg, 
						 cv::Point(position.x, position.y),
						 cv::Point(position.x + velocity.x, position.y + velocity.y),
						 LineColor * speed,
						 LineThickness);
			}

			cv::imshow(winName, outputImg);

			// Press 'ESC' to exit
			if (cv::waitKey(1) == 27)
			{
				break;
			}
		}

		const double drawTime_ms = drawTimer.Elapsed_ms();

		std::cout << "GPU Read Time: " << std::to_string(gpuBufferTime_ms) << "\tDraw Time: " << std::to_string(drawTime_ms) << std::endl;
		deltaTime_s = (gpuBufferTime_ms + drawTime_ms) * 0.01f; // Convert back to seconds
	}

	cv::destroyWindow(winName);

    ///* Deallocate resources */
    
	clReleaseKernel(kernel);
	clReleaseMemObject(positionsBuffer);
	clReleaseMemObject(velocitiesBuffer);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
	return 0;
}