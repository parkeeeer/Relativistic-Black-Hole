#version 410 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragpos;
out vec3 fragnorm;

void main() {
    fragpos = vec3(model * vec4(pos, 1.0));
    fragnorm = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * model * vec4(pos, 1.0f);
}