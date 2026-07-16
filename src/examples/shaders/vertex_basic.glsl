#version 460

// Corresponds to the 'position' field in the Vertex2Df struct.
layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
