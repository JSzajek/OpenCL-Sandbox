__kernel void sepia(__global const uchar4* input,
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

        // Apply sepia filter
        int r = (int)(0.393f * pixel.x + 0.769f * pixel.y + 0.189f * pixel.z);
        int g = (int)(0.349f * pixel.x + 0.686f * pixel.y + 0.168f * pixel.z);
        int b = (int)(0.272f * pixel.x + 0.534f * pixel.y + 0.131f * pixel.z);

        r = min(r, 255);
        g = min(g, 255);
        b = min(b, 255);

        // Clamp values to [0, 255]
        output[index] = (uchar4)((uchar)r, (uchar)g, (uchar)b, pixel.w);
    }
}
