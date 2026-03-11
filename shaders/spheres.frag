#version 410 core
in vec3 fragpos;
in vec3 fragnorm;

out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {

    vec3 ambient = 0.1 * lightColor;


    vec3 norm = normalize(fragnorm);
    vec3 lightDir = normalize(lightPos - fragpos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    fragColor = vec4((ambient + diffuse) * objectColor, 1.0);
}