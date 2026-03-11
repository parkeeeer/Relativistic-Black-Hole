#ifndef RELATAVISTICGRAVITY_WRAPPERS_HPP
#define RELATAVISTICGRAVITY_WRAPPERS_HPP

#define GL_SILENCE_DEPRECATION

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <span>
#include "physics.hpp"

#ifdef __APPLE__
    #include <OpenGL/gl3.h>
#else
    #include <GL/gl.h>
#endif

namespace wrapper {

    struct vertex {
        glm::vec3 position;
        glm::vec3 normal;
    };
    struct Mesh {
        Mesh(const std::vector<vertex>& vertices, const std::vector<unsigned int>& indices) : vcount(indices.size()){
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
            glGenBuffers(1, &ebo);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertex), vertices.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*) 0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, normal));
            glEnableVertexAttribArray(1);

            glBindVertexArray(0);
        }

        void draw() const {
            glBindVertexArray(vao);
            glDrawElements(GL_TRIANGLES, vcount, GL_UNSIGNED_INT, 0);
        }

        ~Mesh() {
            glDeleteVertexArrays(1, &vao);
            glDeleteBuffers(1, &vbo);
            glDeleteBuffers(1, &ebo);
        }

        Mesh(const Mesh&) = delete;
        Mesh& operator=(const Mesh&) = delete;

        Mesh(Mesh&& other) noexcept : vao(other.vao), vbo(other.vbo), ebo(other.ebo), vcount(other.vcount) {
            other.vao = 0;
            other.vbo = 0;
            other.ebo = 0;
        }

        Mesh& operator=(Mesh&& other) noexcept {
            if (this != &other) {
                glDeleteVertexArrays(1, &vao);
                glDeleteBuffers(1, &vbo);
                glDeleteBuffers(1, &ebo);
                vao = other.vao;
                vbo = other.vbo;
                ebo = other.ebo;
                vcount = other.vcount;
            }
            return *this;
        }

    private:
        GLuint vao, vbo, ebo;
        unsigned int vcount;
    };

    class Camera {
        float azimuth;
        float elevation;
        float distance;
        static constexpr glm::vec3 target = glm::vec3(0.0f);
        float fov;
        float aspect;
    public:

        Camera() noexcept :
        azimuth(0.0f),
        elevation(0.0f),
        distance(10.0f),
        fov(60.0f),
        aspect(800.0f / 600.0f)
        {}

        [[nodiscard]] glm::vec3 getPos() const {
            return target + glm::vec3{
                distance * std::cos(elevation) * std::sin(azimuth),
                distance * std::sin(elevation),
                distance * std::cos(elevation) * std::cos(azimuth)
            };
        }

        [[nodiscard]] glm::mat4 getViewMatrix() const {
            return glm::lookAt(getPos(), target, glm::vec3(0.0f, 1.0f, 0.0f));
        }

        [[nodiscard]] glm::mat4 getProjectionMatrix() const {
            return glm::perspective(glm::radians(fov), aspect, 0.1f, 1000.0f);
        }

        void orbit(float dazimuth, float delevation) {
            azimuth += dazimuth;
            elevation = glm::clamp(elevation + delevation, -1.2f, 1.2f);
        }

        void zoom(float delta) {
            distance = glm::max(1.0f, distance - delta);
        }
        [[nodiscard]] glm::vec3 getForward() const {
            return glm::normalize(glm::vec3(0.0f) - getPos());
        }

        [[nodiscard]] glm::vec3 getRight() const {
            glm::vec3 fwd = getForward();
            glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
            glm::vec3 right = glm::cross(fwd, worldUp);
            float len = glm::length(right);
            if (len < 1e-6f) {
                // Forward is nearly parallel to world up — pick a fallback
                right = glm::vec3(1.0f, 0.0f, 0.0f);
            } else {
                right /= len;
            }
            return right;
        }

        [[nodiscard]] glm::vec3 getUp() const {
            return glm::normalize(glm::cross(getRight(), getForward()));
        }

        [[nodiscard]] float getFov() const { return fov; }

        Camera(const Camera&) = delete;
        Camera& operator=(const Camera&) = delete;

        Camera(Camera&& other) noexcept : azimuth(other.azimuth), elevation(other.elevation), fov(other.fov), aspect(other.aspect), distance(other.distance) {}
        Camera& operator=(Camera&& other) noexcept {
            if (this != &other) {
                azimuth = other.azimuth;
                elevation = other.elevation;
                distance = other.distance;
                fov = other.fov;
                aspect = other.aspect;
            }
            return *this;
        }
    };

    class Shader {
        GLuint program;
        static GLuint compileShader(GLenum shadertype, const char* src) {
            GLuint shader = glCreateShader(shadertype);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);
            int success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (success == GL_FALSE) {
                char infolog[512];
                glGetShaderInfoLog(shader, 512, nullptr, infolog);
                std::cerr << "shader compilation error: " << infolog << std::endl;
            }
            return shader;
        }

    public:
        Shader(std::string_view vert_path, std::string_view frag_path) {
            std::ifstream vert_file(vert_path);
            std::ifstream frag_file(frag_path);
            if (!vert_file.is_open()) {
                std::cerr << "vert shader file not found" << std::endl;
                return;
            }
            if (!frag_file.is_open()) {
                std::cerr << "frag shader file not found" << std::endl;
                return;
            }
            std::string vert_src((std::istreambuf_iterator<char>(vert_file)), std::istreambuf_iterator<char>());
            std::string frag_src((std::istreambuf_iterator<char>(frag_file)), std::istreambuf_iterator<char>());
            vert_file.close();
            frag_file.close();
            GLuint vert = compileShader(GL_VERTEX_SHADER, vert_src.c_str());
            GLuint frag = compileShader(GL_FRAGMENT_SHADER, frag_src.c_str());
            program = glCreateProgram();
            glAttachShader(program, vert);
            glAttachShader(program, frag);
            glLinkProgram(program);
            int success;
            glGetProgramiv(program, GL_LINK_STATUS, &success);
            if (!success) {
                char infolog[512];
                glGetProgramInfoLog(program, 512, nullptr, infolog);
                std::cerr << "shader linking error: " << infolog << std::endl;
            }
            glDeleteShader(vert);
            glDeleteShader(frag);
        }

        ~Shader() {
            if (program == 0) return; //nothing to delete
            glDeleteProgram(program);
        }

        void bind() const {
            glUseProgram(program);
        }

        void setMat4(const std::string& name, glm::mat4& mat) const {
            GLint location = glGetUniformLocation(program, name.c_str());
            glProgramUniformMatrix4fv(program, location, 1, GL_FALSE, glm::value_ptr(mat));
        }

        void setVec3(const std::string& name, glm::vec3& vec) const {
            GLint location = glGetUniformLocation(program, name.c_str());
            glProgramUniform3fv(program, location, 1 , glm::value_ptr(vec));
        }

        void setFloat(const std::string& name, float& val) const {
            GLint location = glGetUniformLocation(program, name.c_str());
            glProgramUniform1f(program, location, val);
        }

        void setInt(const std::string& name, int& val) const {
            GLint location = glGetUniformLocation(program, name.c_str());
            glProgramUniform1i(program, location, val);
        }

        Shader(const Shader&) = delete;
        Shader& operator=(const Shader&) = delete;

        Shader(Shader&& other) noexcept : program(other.program) {
            other.program = 0;
        }
        Shader& operator=(Shader&& other) noexcept {
            if (this != &other) {
                program = other.program;
                other.program = 0;
            }
            return *this;
        }

    };

    class Renderer {
        Camera cam;
        Mesh sphere;
        Shader shader;
        Mesh generateSphere() {
            std::vector<vertex> vertices;
            std::vector<unsigned int> indices;
            int stacks = 32;
            int slices = 32;
            float radius = 1.0f;
            for (int i = 0; i <= stacks;++i) {
                float phi = M_PI * i / stacks;
                for (int j = 0; j <= slices;++j) {
                    float theta = 2 * M_PI * j / slices;

                    glm::vec3 pos = {
                        radius * std::sin(phi) * std::cos(theta),
                        radius * std::cos(phi),
                        radius * std::sin(phi) * std::sin(theta)
                    };

                    glm::vec3 normal = glm::normalize(pos);
                    vertices.push_back({pos, normal});
                }
            }

            for (unsigned int i = 0; i <= slices;++i) {
                for (unsigned int j = 0; j <= stacks;++j) {
                    unsigned int up_left = i * (slices + 1) + j;
                    unsigned int up_right = up_left + 1;
                    unsigned int down_left = (i + 1) * (slices + 1) + j;
                    unsigned int down_right = down_left + 1;

                    indices.push_back(up_left);
                    indices.push_back(down_left);
                    indices.push_back(up_right);

                    indices.push_back(down_left);
                    indices.push_back(down_right);
                    indices.push_back(up_right);
                }
            }
            return Mesh(vertices, indices);
        }

    public:
        Renderer() :
            cam(Camera()),
            shader(SHADER_DIR "spheres.vert", SHADER_DIR "spheres.frag"),
            sphere(generateSphere())
        {}
        void draw(std::span<Particle> particles) const {
            shader.bind();
            glm::mat4 projection = cam.getProjectionMatrix();
            glm::mat4 view = cam.getViewMatrix();
            glm::vec3 lightPos = cam.getPos();
            glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);

            shader.setMat4("projection", projection);
            shader.setMat4("view", view);
            shader.setVec3("lightPos", lightPos);
            shader.setVec3("lightColor", lightColor);

            for (auto& particle : particles) {
                glm::mat4 model = particle.getModelMatrix();
                shader.setMat4("model", model);
                shader.setVec3("objectColor", particle.color);
                sphere.draw();
            }
        }

        void draw(const BlackHole& bh) const {
            shader.bind();
            glm::mat4 projection = cam.getProjectionMatrix();
            glm::mat4 view = cam.getViewMatrix();
            glm::mat4 model = bh.getModelMatrix();
            glm::vec3 lightPos = cam.getPos();
            glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
            glm::vec3 objectColor = glm::vec3(0.0f, 0.0f, 0.0f);

            shader.setMat4("projection", projection);
            shader.setMat4("view", view);
            shader.setMat4("model", model);
            shader.setVec3("lightPos", lightPos);
            shader.setVec3("lightColor", lightColor);
            shader.setVec3("objectColor", objectColor);
            sphere.draw();
        }

        Camera& getCamera() noexcept {
            return cam;
        }
    };

    class Raytracer {
        Shader shader;
        GLuint vao;

    public:
        Raytracer() : shader(SHADER_DIR "raytrace.vert", SHADER_DIR "raytrace.frag") {
            glGenVertexArrays(1, &vao);
        }

        ~Raytracer() {
            glDeleteVertexArrays(1, &vao);
        }

        void draw(const Camera& cam, const BlackHole& bh, GLuint sceneTexture) const {
            shader.bind();
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, sceneTexture);
            int temp = 0;
            shader.setInt("sceneTexture", temp);
            glm::vec3 camPos = cam.getPos();
            glm::vec3 camForward = cam.getForward();
            glm::vec3 camRight = cam.getRight();
            glm::vec3 camUp = cam.getUp();
            float FOV = cam.getFov();
            float mass = bh.getMass();
            float spin = bh.getSpin();
            float aspect = 800.0f/600.0f;
            shader.setVec3("camPos", camPos);
            shader.setVec3("camForward", camForward);
            shader.setVec3("camRight", camRight);
            shader.setVec3("camUp", camUp);
            shader.setFloat("fov", FOV);
            shader.setFloat("M", mass);
            shader.setFloat("a", spin);
            shader.setFloat("aspect", aspect);

            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        }

        Raytracer(const Raytracer&) = delete;
        Raytracer& operator=(const Raytracer&) = delete;
    };

    class Framebuffer {
        GLuint fbo;
        GLuint texture;
        GLuint rbo; // renderbuffer for depth

    public:
        Framebuffer(int width, int height) {
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            // color texture
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

            // depth renderbuffer
            glGenRenderbuffers(1, &rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        void bind() const { glBindFramebuffer(GL_FRAMEBUFFER, fbo); }
        void unbind() const { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
        GLuint getTexture() const { return texture; }

        ~Framebuffer() {
            glDeleteFramebuffers(1, &fbo);
            glDeleteTextures(1, &texture);
            glDeleteRenderbuffers(1, &rbo);
        }
    };
}

#endif //RELATAVISTICGRAVITY_WRAPPERS_HPP