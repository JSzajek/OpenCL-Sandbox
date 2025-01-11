
__constant float filter[3][3] = 
{
    { 1 / 16.0f, 2 / 16.0f, 1 / 16.0f },
    { 2 / 16.0f, 4 / 16.0f, 2 / 16.0f },
    { 1 / 16.0f, 2 / 16.0f, 1 / 16.0f }
};

__kernel void blur_img(__global const uchar4* input,
                       __global float* filter,
                       int filter_size,
                       int width,
                       int height,
                       __global uchar4* output) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Half size of the filter (used for offset calculations)
    int half_size = filter_size / 2;

    if (x < width && y < height) 
    {
        float4 sum = (float4)(0.0f);
        float weight_sum = 0.0f;

        // Apply the filter
        for (int fx = -half_size; fx <= half_size; fx++) 
        {
            for (int fy = -half_size; fy <= half_size; fy++) 
            {
                int nx = clamp(x + fx, 0, width - 1);  // Clamp to valid range
                int ny = clamp(y + fy, 0, height - 1); // Clamp to valid range
                int filter_index = (fy + half_size) * filter_size + (fx + half_size);
                int pixel_index = ny * width + nx;

                uchar4 pixel = input[pixel_index];
                float weight = filter[filter_index];

                sum += (float4)(pixel.x, pixel.y, pixel.z, pixel.w) * weight;
                weight_sum += weight;
            }
        }

        // Normalize and store the result
        int index = y * width + x;
        output[index] = (uchar4)((uchar)(sum.x / weight_sum),
                                 (uchar)(sum.y / weight_sum),
                                 (uchar)(sum.z / weight_sum),
                                 (uchar)(sum.w / weight_sum));
    }
}
