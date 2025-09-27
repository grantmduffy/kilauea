#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;

layout(binding = 0) uniform Uniforms {
    mat4 camera;
    float time;
} uniforms;

void main() {
    gl_Position = uniforms.camera * vec4(inPosition, 0.0, 1.0);
    fragTexCoord = inTexCoord;
}
