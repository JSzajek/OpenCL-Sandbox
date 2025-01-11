__kernel void halftone(__global const uchar4* input,
                       float dot_radius,
                       float scale,
                       int width,
                       int height,
                       __global uchar4* output) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) 
    {
        // Calculate normalized grid position
        float2 grid_pos = (float2)(x, y) / scale;
        int cell_center_x = round(grid_pos.x) * scale;
        int cell_center_y = round(grid_pos.y) * scale;

        // Ensure the center is within bounds
        cell_center_x = clamp(cell_center_x, 0, width - 1);
        cell_center_y = clamp(cell_center_y, 0, height - 1);

        // Read the intensity at the center of the grid cell
        uchar4 center_pixel = input[cell_center_y * width + cell_center_x];
        float intensity = (center_pixel.x + center_pixel.y + center_pixel.z) / 3.0f / 255.0f;

        // Map intensity to dot size
        float dot_size = intensity * dot_radius;

        // Compute distance from current pixel to the grid center
        float dist = length((float2)(x, y) - (float2)(cell_center_x, cell_center_y));

        // Determine pixel color based on distance to dot
        uchar4 output_pixel;
        if (dist < dot_size) 
        {
            // Inside the dot: use the color of the center pixel
            output_pixel = center_pixel;
        }
        else 
        {
            // Outside the dot: set to white
            output_pixel = (uchar4)(255, 255, 255, center_pixel.w);
        }

        // Write the output pixel
        output[y * width + x] = output_pixel;
    }
}
