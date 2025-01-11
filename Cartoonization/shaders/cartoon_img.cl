__kernel void cartoonize(__global const uchar4* input,
                         __global const float* kernel_x,
                         __global const float* kernel_y,
                         int quantization_levels,
                         float edge_threshold,
                         int width,
                         int height,
                         __global uchar4* output) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int sobel_size = 3; // Sobel kernel size is 3x3

    if (x < width && y < height) 
    {
        int index = y * width + x;
        uchar4 pixel = input[index];

        // Convert to grayscale
        float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

        // Sobel edge detection
        float gx = 0.0f;
        float gy = 0.0f;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = x + dx;
                int ny = y + dy;

                if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                    continue;

                int neighbor_idx = ny * width + nx;
                uchar4 neighbor_pixel = input[neighbor_idx];
                float neighbor_gray = 0.299f * neighbor_pixel.x + 0.587f * neighbor_pixel.y + 0.114f * neighbor_pixel.z;

                int kernel_idx = (dy + 1) * sobel_size + (dx + 1);
                gx += neighbor_gray * kernel_x[kernel_idx];
                gy += neighbor_gray * kernel_y[kernel_idx];
            }
        }

        float edge_magnitude = sqrt((float)(gx * gx + gy * gy));

        // Color quantization
        uchar4 quantized_pixel = pixel;
        quantized_pixel.x = (uchar)((pixel.x / quantization_levels) * quantization_levels);
        quantized_pixel.y = (uchar)((pixel.y / quantization_levels) * quantization_levels);
        quantized_pixel.z = (uchar)((pixel.z / quantization_levels) * quantization_levels);

        // Combine edge detection and quantization
        if (edge_magnitude > edge_threshold)
        {
            // Strong edge, paint black
            output[index] = (uchar4)(0, 0, 0, 255);
        }
        else
        {
            // No edge, use quantized color
            output[index] = quantized_pixel;
        }
    }
}
