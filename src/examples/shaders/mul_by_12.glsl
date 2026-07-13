#version 460

// Will affect the index we obtain from `gl_GlobalInvocationID`.
// Each workgroup will have a local size of 64 work units, so the local size is set to 64.
// `y` and `z` are set to `1` because our input buffer is one-dimensional.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    // Get index of element this unit will work on.
    uint index = gl_GlobalInvocationID.x;

    // Actual operation.
    buf.data[index] *= 12;
}
