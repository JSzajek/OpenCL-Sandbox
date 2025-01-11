__kernel void simulate(__global float2* positions,
                       __global float2* velocities,
                       __global float2* accelerations,
                       const float dt,
                       const float separation_radius,
                       const float alignment_radius,
                       const float cohesion_radius,
                       const float max_speed,
                       const float max_force,
                       const float bounceFactor,
                       float4 bounds) 
{
    int i = get_global_id(0);
    int NumBoids = get_global_size(0);

    // Read current position and velocity
    float2 pos = positions[i];
    float2 vel = velocities[i];
    float2 acc = accelerations[i];

    // Behavior calculations
    float2 separation   = (float2)(0.0f, 0.0f);
    float2 alignment    = (float2)(0.0f, 0.0f);
    float2 cohesion     = (float2)(0.0f, 0.0f);

    int separation_count = 0;
    int alignment_count = 0;
    int cohesion_count = 0;

    // Iterate through all other boids for behaviors
    for (int j = 0; j < get_global_size(0); j++) 
    {
        if (i == j) 
            continue;  // Skip self

        // Get position of other boid
        float2 other_pos = positions[j];
        float2 diff = other_pos - pos;
        float distance = length(diff);

        // Separation: Avoid other boids within separation radius
        if (distance < separation_radius)
        {
            float forceStrength = 1.0f / (distance * distance);
            separation += normalize(diff) * forceStrength;
            separation_count++;
        }

        // Alignment: Align with nearby boids within alignment radius
        if (distance < alignment_radius)
        {
            alignment += velocities[j];
            alignment_count++;
        }

        // Cohesion: Move towards the average position of nearby boids
        if (distance < cohesion_radius)
        {
            cohesion += other_pos;
            cohesion_count++;
        }
    }

    // Compute average behaviors (avoid division by zero)
    if (separation_count > 0)
    {
        separation /= separation_count;
    }
    if (alignment_count > 0)
    {
        alignment /= alignment_count;
    }
    if (cohesion_count > 0)
    {
        cohesion /= cohesion_count;
    }

    // Apply behaviors
    separation = normalize(separation) * max_force;
    alignment = normalize(alignment) * max_force;
    cohesion = normalize(cohesion - pos) * max_force;

    // Update acceleration: sum all the behaviors
    acc += separation + alignment + cohesion;

    // Limit the acceleration
    acc = normalize(acc) * max_force;

    // Update velocity based on acceleration
    vel += acc * dt;

    // Limit velocity to max speed
    if (length(vel) > max_speed) 
    {
        vel = normalize(vel) * max_speed;
    }

    // Update position based on velocity
    pos += vel * dt;

    // Handle boundary collisions
    float minX = bounds.x;
    float maxX = bounds.y;
    float minY = bounds.z;
    float maxY = bounds.w;

    if (pos.x < minX)
    {
        pos.x = maxX;
    }
    if (pos.x > maxX)
    {
        pos.x = minX;
    }
    if (pos.y < minY)
    {
        pos.y = maxY;
    }
    if (pos.y > maxY)
    {
        pos.y = minY;
    }
    
    // Write updated position, velocity, and acceleration back to buffers
    positions[i] = pos;
    velocities[i] = vel;
    accelerations[i] = acc;
}
