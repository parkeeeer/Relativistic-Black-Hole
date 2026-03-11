#version 410 core

out vec2 uv;

void main() {
    vec2 positions[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
    );

    uv = positions[gl_VertexID] * 0.5 + 0.5; // 0 to 1
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}