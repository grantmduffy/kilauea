#version 450

layout(location = 0) in vec2 vertexPosition;
layout(location = 1) in vec3 vertexColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 camera;
    float time;
} ubo;

layout(location = 0) out vec3 fragColor;

void main() {
    // Apply camera transformation to vertex position
    vec4 worldPos = vec4(vertexPosition, 0.0, 1.0);
    gl_Position = ubo.camera * worldPos;
    
    // Pass color to fragment shader, optionally modulated by time
    fragColor = vertexColor * (0.8 + 0.2 * sin(ubo.time));
}
