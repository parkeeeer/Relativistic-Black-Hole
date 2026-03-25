#include <iostream>


#include "renderer.hpp"

#define GL_SILENCE_DEPRECATION

#ifdef __APPLE__
    #include <OpenGL/gl3.h>
#else
    #include <GL/gl.h>
#endif

#include <GLFW/glfw3.h>

#include <iostream>

#include "renderer.hpp"
#include "physics.hpp"



#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

using namespace std;

int main() {
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window;
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Black Hole", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);

    glEnable(GL_DEPTH_TEST);

    float M = 1.0f, r = 6.0f;
    float ut = 1.0f / sqrt(1 - 3*M/r);
    float uph = sqrt(M / (r*r*r)) / sqrt(1 - 3*M/r);

    BlackHole bh(1.0f, 0.0f);
    Particle p(
        glm::vec4(0, 6, M_PI/2, 0),  // position
        glm::vec4(ut, 0, 0, uph),     // 4-velocity
        2.0f, 1.0f, glm::vec3(0,1,0), bh
    );


    vector<Particle> particles;
    particles.push_back(p);

    wrapper::Renderer rend;
    wrapper::Raytracer rt;
    wrapper::Framebuffer fb(WINDOW_WIDTH, WINDOW_HEIGHT);

    glfwSetWindowUserPointer(window, &rend);

    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(w, true);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        static double last_x, last_y;
        double dx = x - last_x;
        double dy = y - last_y;
        last_x = x;
        last_y = y;
        if (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            auto r = static_cast<wrapper::Renderer*>(glfwGetWindowUserPointer(w));
            r->getCamera().orbit(dx * .01f, dy * .01f);
        }
    });

    glfwSetScrollCallback(window, [](GLFWwindow* w, double x, double y) {
        auto r = static_cast<wrapper::Renderer*>(glfwGetWindowUserPointer(w));
        r->getCamera().zoom(y * .05f);
    });


    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (auto& p : particles) {
            integrate(p, bh, 0.1f);
        }
        fb.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        rend.draw(particles);
        rend.draw(bh);
        fb.unbind();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        rt.draw(rend.getCamera(), bh, fb.getTexture());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}