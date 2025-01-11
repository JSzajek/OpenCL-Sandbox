#pragma once

#include "Cl/cl.h"

class OpenCLUtils 
{
public:
    /// <summary>
	/// Find a GPU or CPU associated with the first available platform
	/// The `platform` structure identifies the first platform identified by the
	/// OpenCL runtime.A platform identifies a vendor's installation, so a system
	/// may have an NVIDIA platform and an AMD platform.
	/// 
	/// The `device` structure corresponds to the first accessible device
	/// associated with the platform.Because the second parameter is
	/// `CL_DEVICE_TYPE_GPU`, this device must be a GPU.
    /// </summary>
    /// <returns></returns>
    static cl_device_id create_device();

    /// <summary>
    /// Create program from a file and compile it. 
    /// </summary>
    /// <param name="ctx"></param>
    /// <param name="dev"></param>
    /// <param name="filename"></param>
    /// <returns></returns>
    static cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);

    static cl_mem create_input_buffer(cl_context context, void* dataPtr, size_t dataSize);

    static cl_mem create_output_buffer(cl_context context, size_t dataSize);
};