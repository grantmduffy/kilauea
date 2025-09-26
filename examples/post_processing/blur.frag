#version 450

#define size 20

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform sampler2D texSampler;

void main() {
    vec2 texelSize = 1.0 / textureSize(texSampler, 0);
    
    outColor = texture(texSampler, fragTexCoord);
    for (int i = -size / 2; i <= size / 2; i++){
        outColor += texture(texSampler, fragTexCoord + vec2(i, 0.0) * texelSize);
    }
    for (int i = -size / 2; i <= size / 2; i++){
        outColor += texture(texSampler, fragTexCoord + vec2(0.0, i) * texelSize);
    }
    outColor /= 2 * size + 1;
    outColor /= 3;
}
