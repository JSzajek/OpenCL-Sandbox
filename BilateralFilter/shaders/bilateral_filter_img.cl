__kernel void filter(__global const uchar4* input,
                     int filter_size,
                     float spatial_sigma, 
                     float intensity_sigma,
                     int width,
                     int height,
                     __global uchar4* output) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) 
    {
        int index = y * width + x;
        uchar4 pixel = input[index];

        // Initialize sums for weighted values and weights
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_w = 0.0f;

        // Loop through the neighborhood
        for (int ky = -filter_size; ky <= filter_size; ky++)
        {
            for (int kx = -filter_size; kx <= filter_size; kx++)
            {
                int nx = x + kx;
                int ny = y + ky;

                // Ensure the neighbor is within bounds
                if (nx >= 0 && ny >= 0 && nx < width && ny < height) 
                {
                    int neighbor_index = ny * width + nx;
                    uchar4 neighbor = input[neighbor_index];

                    // Compute spatial distance (Euclidean distance between pixel locations)
                    float spatial_dist = (kx * kx + ky * ky) / (2.0f * spatial_sigma * spatial_sigma);

                    // Compute intensity difference (Euclidean distance between pixel colors)
                    float intensity_dist = (float)((pixel.x - neighbor.x) * (pixel.x - neighbor.x) +
                                                   (pixel.y - neighbor.y) * (pixel.y - neighbor.y) +
                                                   (pixel.z - neighbor.z) * (pixel.z - neighbor.z)) /
                                                   (2.0f * intensity_sigma * intensity_sigma);

                    // Compute spatial and intensity weights
                    float spatial_weight = exp(-spatial_dist);
                    float intensity_weight = exp(-intensity_dist);
                    float weight = spatial_weight * intensity_weight;

                    // Accumulate weighted color values
                    sum_r += weight * neighbor.x;
                    sum_g += weight * neighbor.y;
                    sum_b += weight * neighbor.z;
                    sum_w += weight;
                }
            }
        }

        // Normalize the result by dividing by the sum of weights
        if (sum_w > 0.0f) 
        {
            sum_r /= sum_w;
            sum_g /= sum_w;
            sum_b /= sum_w;
        }

        // Clamp the resulting values to ensure valid pixel values (0-255 range)
        sum_r = clamp(sum_r, 0.0f, 255.0f);
        sum_g = clamp(sum_g, 0.0f, 255.0f);
        sum_b = clamp(sum_b, 0.0f, 255.0f);

        // Write the result to the output image
        output[index] = (uchar4)(sum_r, sum_g, sum_b, pixel.w);
    }
}
