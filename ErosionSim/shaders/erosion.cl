__kernel void simulate(__global float* height_map,
                       __global float* water_map,
                       __global float* sediment_map,
                       const float erosion_rate,   
                       const float deposition_rate,
                       const float flow_speed,     
                       const int width,
                       const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width + x;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) 
        return;

    // Water flow directions
    float flow_x = 0.0f;
    float flow_y = 0.0f;

    // Calculate height differences with neighbors
    float height_center = height_map[idx] + water_map[idx];
    for (int dx = -1; dx <= 1; dx++) 
    {
        for (int dy = -1; dy <= 1; dy++) 
        {
            if (dx == 0 && dy == 0) 
                continue;

            int neighbor_idx = (y + dy) * width + (x + dx);
            float neighbor_height = height_map[neighbor_idx] + water_map[neighbor_idx];

            float delta = height_center - neighbor_height;
            if (delta > 0) 
            {
                flow_x += dx * delta;
                flow_y += dy * delta;
            }
        }
    }

    // Normalize flow direction
    float magnitude = sqrt(flow_x * flow_x + flow_y * flow_y);
    if (magnitude > 0) 
    {
        flow_x /= magnitude;
        flow_y /= magnitude;
    }
    else 
    {
        flow_x = 0.0f;
        flow_y = 0.0f;
    }

    // Erode terrain and add sediment
    float erosion = erosion_rate * magnitude;
    height_map[idx] -= erosion;
    sediment_map[idx] += erosion;

    // Move sediment based on flow
    int target_x = clamp((int)(x + flow_x * flow_speed), 0, width - 1);
    int target_y = clamp((int)(y + flow_y * flow_speed), 0, height - 1);
    int target_idx = target_y * width + target_x;

    float sediment_transfer = sediment_map[idx] * flow_speed;
    sediment_map[idx] -= sediment_transfer;
    sediment_map[target_idx] += sediment_transfer;

    // Deposit sediment
    float deposition = deposition_rate * sediment_map[idx];
    sediment_map[idx] -= deposition;
    height_map[idx] += deposition;

    // Prevent over or underflow
    height_map[idx] = fmax(height_map[idx], 0.0f);
    water_map[idx] = fmax(water_map[idx], 0.0f);
    sediment_map[idx] = fmax(sediment_map[idx], 0.0f);
}
