__kernel void histo_grayscale(__global const uchar4* input,
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

        // Convert to grayscale intensity using the standard formula
        float intensity = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

        // Find the min and max intensities in the image (pre-compute and pass as input, or compute in the kernel)
        // For simplicity, assume we have precomputed min_intensity and max_intensity
        float min_intensity = 0.0f;  // Example min intensity
        float max_intensity = 255.0f;  // Example max intensity

        // Apply histogram equalization formula
        float normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity);
        uchar norm_pixel = (uchar)(max(normalized_intensity * 255.0f, 0.0f));

        // Use the equalized intensity for the output pixel
        output[index] = (uchar4)(norm_pixel, norm_pixel, norm_pixel, 255);
    }
}
