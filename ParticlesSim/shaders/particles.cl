__kernel void simulate(__global float2* positions,
                       __global float2* velocities,
                       float dt,
                       float2 gravity,
                       float4 bounds,
                       float bounce_factor) 
{
    int id = get_global_id(0);
    
    // Load particle position and velocity
    float2 pos = positions[id];
    float2 vel = velocities[id];

    // Apply gravity
    vel += gravity * dt;

    // Update position
    pos += vel * dt;

    // Handle boundary collisions
    float minX = bounds.x;
    float maxX = bounds.y;
    float minY = bounds.z;
    float maxY = bounds.w;
    if (pos.x < minX)
    {
        pos.x = minX;
        vel.x = -vel.x * bounce_factor;
    }
    else if (pos.x > maxX)
    {
        pos.x = maxX;
        vel.x = -vel.x * bounce_factor;
    }

    if (pos.y < minY)
    {
        pos.y = minY;
        vel.y = -vel.y * bounce_factor;
    }
    else if (pos.y > maxY) {
        pos.y = maxY;
        vel.y = -vel.y * bounce_factor;
    }

    // Store updated values
    positions[id] = pos;
    velocities[id] = vel;
}
