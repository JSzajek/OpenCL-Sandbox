__kernel void oil_paint(__global const uchar4* input,
                        int radius,
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

        // Get the size of the local window
        int window_size = (2 * radius + 1) * (2 * radius + 1);

        // Get the coordinates of the current pixel
        int2 coord = (int2)(x, y);

        // Define bins for intensity histogram
        const int NUM_BINS = 256;
        int histogram[NUM_BINS] = { 0 };

        // Variables to store most frequent intensity and corresponding color
        int max_frequency = 0;
        int mode_intensity = 0;
        float3 avg_color = (float3)(0.0f);

        // Define a square neighborhood around the current pixel
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                // Neighbor coordinates
                int nx = x + dx;
                int ny = y + dy;

                // Clamp coordinates to image bounds
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) 
                    continue;

                // Compute the 1D index for the neighbor
                int neighbor_idx = ny * width + nx;

                // Read the neighbor pixel (RGBA as uchar4)
                uchar4 neighbor_pixel = input[neighbor_idx];

                // Convert to grayscale intensity (0–255)
                int intensity = (int)(0.299f * neighbor_pixel.x +
                                      0.587f * neighbor_pixel.y +
                                      0.114f * neighbor_pixel.z);

                // Update the histogram
                histogram[intensity]++;

                // Check if this intensity has the highest frequency
                if (histogram[intensity] > max_frequency) 
                {
                    max_frequency = histogram[intensity];
                    mode_intensity = intensity;
                    avg_color = (float3)(neighbor_pixel.x, neighbor_pixel.y, neighbor_pixel.z);
                }
            }
        }

        // Set the output pixel to the most frequent color
        uchar avg_r = (uchar)max(avg_color.x, 0.0f);
        uchar avg_g = (uchar)max(avg_color.y, 0.0f);
        uchar avg_b = (uchar)max(avg_color.z, 0.0f);
        output[index] = (uchar4)(avg_r, avg_g, avg_b, 255);
    }
}
