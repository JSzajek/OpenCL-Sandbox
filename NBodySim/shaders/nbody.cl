__kernel void simulate(__global float2* positions,
                       __global float2* velocities,
                       __global float2* accelerations,
                       __global float* radii,
                       __global float* masses,
                       float dt,
                       int numParticles,
                       float gravitationalConstant,
                       float bounceFactor,
                       float4 bounds) 
{
    int i = get_global_id(0);
    
    float2 Pos = positions[i];
    float2 Acc = (float2)(0.0f, 0.0f);

    float Radius = radii[i];
    float Mass = masses[i];

    // Compute forces exerted by all other particles
    for (int j = 0; j < numParticles; j++) 
    {
        if (i == j) 
            continue;

        float2 otherPos = positions[j];
        float2 diff = otherPos - Pos;
        float distanceSq = diff.x * diff.x + diff.y * diff.y + 1e-10f;
        float distance = sqrt(distanceSq);

        // Softened gravitational force with radius consideration
        float otherRadius = radii[j];
        float otherMass = masses[j];

        float distWithSoftening = distance + (Radius + otherRadius); // Add radii for collision threshold
        float force = gravitationalConstant * Mass * otherMass / (distanceSq + distWithSoftening * distWithSoftening); // Softening factor
        Acc += force * diff / distance; // Normalize the direction and scale by force
    }

    // Update acceleration
    accelerations[i] = Acc;

    // Update velocity and position
    velocities[i] += Acc * dt;
    positions[i] += velocities[i] * dt;

    // Handle boundary collisions
    float minX = bounds.x;
    float maxX = bounds.y;
    float minY = bounds.z;
    float maxY = bounds.w;

    // Constrain to bounding area
    const float buffer = 0.1; // Small buffer to prevent sticking
    if (positions[i].x < minX) 
    {
        velocities[i].x *= -bounceFactor; // Reflect velocity
        positions[i].x = minX + buffer; // Adjust position inward
    }
    else if (positions[i].x > maxX) 
    {
        velocities[i].x *= -bounceFactor; // Reflect velocity
        positions[i].x = maxX - buffer; // Adjust position inward
    }

    if (positions[i].y < minY) 
    {
        velocities[i].y *= -bounceFactor; // Reflect velocity
        positions[i].y = minY + buffer; // Adjust position inward
    }
    else if (positions[i].y > maxY) 
    {
        velocities[i].y *= -bounceFactor; // Reflect velocity
        positions[i].y = maxY - buffer; // Adjust position inward
    }
}
