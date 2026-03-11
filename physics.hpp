#ifndef RELATAVISTICGRAVITY_PHYSICS_HPP
#define RELATAVISTICGRAVITY_PHYSICS_HPP

#include <cmath>

#include <glm/glm.hpp>



class BlackHole {
    float mass;
    float spin;

public:
    BlackHole(float mass, float spin) : mass(mass), spin(spin) {}

    float getEventHorizon() const {
        return mass + std::sqrt(mass * mass - spin * spin);
    }

    [[nodiscard]] glm::mat4 getModelMatrix() const {
        return glm::scale(glm::mat4(1.0f), glm::vec3(getEventHorizon()));
    }

    [[nodiscard]] float getMass() const { return mass; }
    [[nodiscard]] float getSpin() const { return spin; }
};

glm::vec4 dpos(const glm::vec4& pos, float E, float Lz, float Q, float mu, bool R_pos, bool theta_pos, const BlackHole& bh) {
    float r = pos.y, theta = pos.z;
    float M = bh.getMass(), a = bh.getSpin();
    float sigma = r*r + a*a*std::cos(theta)*std::cos(theta);
    float delta = r*r - 2*M*r + a*a;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);
    float sin2 = sin_theta * sin_theta;

    float P = E*(r*r + a*a) - a*Lz;
    float R = P*P - delta*(mu*mu*r*r + (Lz - a*E)*(Lz - a*E) + Q);
    float Theta = Q - cos_theta*cos_theta*(a*a*(mu*mu - E*E) + (Lz*Lz/sin2));

    return {
        (-a*(a*E*sin2 - Lz) + (r*r + a*a)/delta * P) / sigma,
        (R_pos ? 1.0f : -1.0f) * std::sqrt(std::abs(R)) / sigma,
        (theta_pos ? 1.0f : -1.0f) * std::sqrt(std::abs(Theta)) / sigma,
        (-(a*E - Lz/sin2) + a/delta * P) / sigma
    };
}

struct Particle {
    glm::vec4 pos;  //uses Boyer-Liquidist for ease with Kerr metric, use getCartesionPos for cartesian
    glm::vec4 fourvel;
    float radius;
    float mass;
    glm::vec3 color;

    //conserved qualitied for Kerr!!!
    float mu;
    float E;
    float Lz;
    float Q;

    bool R_pos = true;
    bool theta_pos = true;

    Particle(glm::vec4 pos, glm::vec4 fourvel, float radius, float mass, glm::vec3 color) : pos(pos), fourvel(fourvel), radius(radius), mass(mass), color(color) {}


    Particle(glm::vec4 pos, glm::vec4 fourvel, float radius, float mass, glm::vec3 color, const BlackHole& bh) : pos(pos), fourvel(fourvel), radius(radius), mass(mass), color(color) {
        float r = pos.y, theta = pos.z;
        float M = bh.getMass(), a = bh.getSpin();
        float sigma = r * r + a * a * std::cos(theta) * std::cos(theta);
        float delta = r * r - 2.0f * M * r + a * a; //Schwarzchild radius = 2 * M
        float sin2 = std::sin(theta) * std::sin(theta);

        float g_tt = -(1.0f- 2*M*r/sigma);
        float g_rr = sigma / delta;
        float g_thth = sigma;
        float g_phph = (r*r + a*a + 2*M*r*a*a*sin2/sigma) * sin2;
        float g_tph = -2*M*r*a*sin2/sigma;

        float ut = fourvel.x, ur = fourvel.y, uth = fourvel.z, uph = fourvel.w;

        mu = std::sqrt(-(g_tt*ut*ut + g_rr*ur*ur + g_thth*uth*uth + g_phph*uph*uph + 2*g_tph*ut*uph));
        E  = -(g_tt*ut + g_tph*uph);
        Lz =  g_phph*uph + g_tph*ut;
        Q  =  g_thth*uth*uth + std::cos(theta)*std::cos(theta) * (a*a*(mu*mu - E*E) + (Lz/std::sin(theta))*(Lz/std::sin(theta)));
    }
    [[nodiscard]] glm::vec3 getCartesianPos(float spin = 0.0f) const {
        float r = pos.y, theta = pos.z, phi = pos.w;
        float rho = std::sqrt(r * r + spin * spin);
        return {
            rho * std::sin(theta) * std::cos(phi),
            r * std::cos(theta),
            rho * std::sin(theta) * std::sin(phi)
        };
    }
    [[nodiscard]] glm::mat4 getModelMatrix() const {
        glm::vec3 cartpos = getCartesianPos();
        glm::mat4 model = glm::translate(glm::mat4(1.0f), cartpos);
        return glm::scale(model, glm::vec3(radius));
    }
};

void integrate(Particle& p, const BlackHole& bh, float dt) {
    auto f = [&](glm::vec4& pos){return dpos(pos, p.E, p.Lz, p.Q, p.mu, p.R_pos, p.theta_pos, bh);};

    float r = p.pos.y, theta = p.pos.z;
    float M = bh.getMass(), a = bh.getSpin();
    float sigma = r*r + a*a*std::cos(theta)*std::cos(theta);
    float delta = r*r - 2*M*r + a*a;
    float sin2 = std::sin(theta)*std::sin(theta);
    float cos2 = std::cos(theta)*std::cos(theta);

    float P = p.E*(r*r + a*a) - a*p.Lz;
    float R = P*P - delta*(p.mu*p.mu*r*r + (p.Lz - a*p.E)*(p.Lz - a*p.E) + p.Q);
    float Theta = p.Q - cos2*(a*a*(p.mu*p.mu - p.E*p.E) + (p.Lz*p.Lz/sin2));

    if (R < 0) p.R_pos = !p.R_pos;
    if (Theta < 0) p.theta_pos = !p.theta_pos;

    glm::vec4 k1 = f(p.pos);
    glm::vec4 temp = p.pos + .5f*dt*k1;
    glm::vec4 k2 = f(temp);
    temp = p.pos + .5f*dt*k2;
    glm::vec4 k3 = f(temp);
    temp = p.pos + dt*k3;
    glm::vec4 k4 = f(temp);

    p.pos += dt/6.0f * (k1 + 2.0f*k2 + 2.0f*k3 + k4);
}







#endif //RELATAVISTICGRAVITY_PHYSICS_HPP