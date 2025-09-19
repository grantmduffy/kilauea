#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform sampler2D texSampler;

void main() {
    float pixelSize = 3.0 * 1.0 / textureSize(texSampler, 0).x;
    vec4 result = vec4(0.0);
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            result += texture(texSampler, fragTexCoord + vec2(x, y) * pixelSize);
        }
    }
    outColor = result / 25.0;
}
