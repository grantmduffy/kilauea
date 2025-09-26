#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;

layout(std140, set = 0, binding = 0) uniform UniformBufferObject {
    mat4 camera;
    float time;
} ubo;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.camera * vec4(inPos, 0.0, 1.0);
    fragColor = inColor;
}
