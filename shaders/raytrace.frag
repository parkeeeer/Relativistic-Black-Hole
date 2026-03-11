#version 410 core

//this is the worst thing i have ever had to write
//i hope you enjoy

// ============================================================
//  Kerr Black Hole Raytracer -- Kerr-Schild Cartesian Coordinates
//
//  No coordinate singularity at the poles.
//
//  The Kerr-Schild form of the metric is:
//    g = eta + f * l (x) l
//
//  where eta is the flat Minkowski metric diag(-1,1,1,1),
//  f is a scalar function, and l is a null 1-form.
//
//  Everything is in Cartesian coords (t, x, y, z) with y as the
//  polar/spin axis. No sin(theta) anywhere, no pole problems.
//
//  We integrate the second-order geodesic equation:
//    d^2 x^i / dlambda^2 = acceleration from curved spacetime
//
//  tracking position (x,y,z) and spatial velocity (vx,vy,vz).
//  The time component vt is recovered from the null condition
//  (light travels on null geodesics: g_uv * v^u * v^v = 0)
//  at each integration step.
// ============================================================

in vec2 uv;
out vec4 fragColor;

uniform vec3  camPos;
uniform vec3  camForward;
uniform vec3  camRight;
uniform vec3  camUp;
uniform float fov;
uniform float aspect;
uniform float M;           // black hole mass
uniform float a;           // spin parameter, |a| <= M
uniform sampler2D sceneTexture;

const int   MAX_STEPS = 2000;
const float ESCAPE_R2 = 150.0 * 150.0;  // squared escape radius
const float PI        = 3.14159265359;


// ============================================================
// Compute the Boyer-Lindquist radius r from Cartesian position.
//
// In Kerr-Schild coords, r is defined implicitly by:
//   r^4 - (x^2+y^2+z^2 - a^2)*r^2 - a^2*z_pole^2 = 0
//
// where z_pole is the component along the spin axis (= p.y here).
// Solving this quadratic in r^2:
//   r^2 = 0.5 * (w2 + sqrt(w2^2 + 4*a^2*z_pole^2))
//   where w2 = x^2+y^2+z^2 - a^2
// ============================================================
float kerr_r(vec3 p) {
    float a2 = a * a;
    float w2 = dot(p, p) - a2;
    float z2 = p.y * p.y;          // y is the spin/polar axis
    float r2 = 0.5 * (w2 + sqrt(max(w2 * w2 + 4.0 * a2 * z2, 0.0)));
    return sqrt(max(r2, 1e-10));
}


// ============================================================
// Compute the geodesic acceleration (spatial part only).
//
// Given position p = (x, y, z) and 4-velocity v4 = (vt, vx, vy, vz),
// returns the spatial acceleration vec3(ax, ay, az).
//
// The metric perturbation is h_uv = f * l_u * l_v where:
//
//   f = 2*M*r^3 / (r^4 + a^2 * z_pole^2)
//
//   l = (1, lx, ly, lz) with:
//     lx = (r*px + a*pz) / (r^2 + a^2)     [equatorial component]
//     ly = py / r                             [polar component]
//     lz = (r*pz - a*px) / (r^2 + a^2)     [equatorial component]
//
// The acceleration comes from the geodesic equation:
//   a^i = -Christoffel^i_uv * v^u * v^v
//
// Which we compute by differentiating h_uv directly rather than
// building the full Christoffel symbol table. We need:
//   - df/dx_i  (gradient of f)
//   - dl_j/dx_i  (3x3 Jacobian of the l vector)
//
// Both require dr/dx_i (gradient of r), which comes from
// implicit differentiation of the r^4 equation:
//   dr/dx_i = (x_i * r^2 + a^2 * z_pole * delta_{y,i}) / (r * D)
//   where D = sqrt((rho^2 - a^2)^2 + 4*a^2*z_pole^2)
//
// We then assemble two contracted quantities:
//   Q_beta = (v^u * d_u h_{beta,v}) * v^v
//   S_beta = 0.5 * (d_beta h_{u,v}) * v^u * v^v
//
// And the acceleration is:
//   a^i = -(Q_i - S_i) + f * l^i * (l^beta * (Q_beta - S_beta))
//
// where l^beta = (-1, lx, ly, lz) with the time index flipped.
// ============================================================
vec3 geodesic_accel(vec3 p, vec4 v4) {
    float a2 = a * a;

    // --- Compute r and dr/dx_i ---
    float rho2 = dot(p, p);
    float z_pole = p.y;
    float z2 = z_pole * z_pole;
    float w2 = rho2 - a2;
    float disc = max(w2 * w2 + 4.0 * a2 * z2, 0.0);
    float D = sqrt(disc);
    float r2 = max(0.5 * (w2 + D), 1e-10);
    float r = sqrt(r2);
    float r4 = r2 * r2;

    // Gradient of r with respect to (x, y, z)
    // dr/dx = x*r / D,  dr/dy = y*(r^2+a^2) / (r*D),  dr/dz = z*r / D
    // (y gets the extra a^2 term because y is the spin axis)
    float inv_rD = 1.0 / max(r * D, 1e-20);
    vec3 dr_dx = vec3(
    p.x * r2 * inv_rD,              // dr/dx
    p.y * (r2 + a2) * inv_rD,       // dr/dy (spin axis, extra a^2 term)
    p.z * r2 * inv_rD               // dr/dz
    );

    // --- f and its gradient ---
    // f = 2*M*r^3 / (r^4 + a^2 * z_pole^2)
    float denom = r4 + a2 * z2;
    float f = 2.0 * M * r * r2 / max(denom, 1e-20);

    // df/dx_i via quotient rule on f = 2M * r^3 / denom
    // numerator of quotient rule:
    //   d(r^3)/dx_i * denom - r^3 * d(denom)/dx_i
    // where d(r^3)/dx_i = 3*r^2 * dr/dx_i
    // and   d(denom)/dx_i = 4*r^3 * dr/dx_i + 2*a^2*z_pole * dz/dx_i
    // and   dz/dx_i = (0, 1, 0) since z_pole = p.y
    float denom2 = denom * denom;
    float twoM = 2.0 * M;
    vec3 dz_dx = vec3(0.0, 1.0, 0.0);

    vec3 df_dx;
    for (int i = 0; i < 3; i++) {
        float dr_i = dr_dx[i];
        float dz_i = dz_dx[i];
        float num = 3.0*r2*dr_i * denom - r*r2 * (4.0*r*r2*dr_i + 2.0*a2*z_pole*dz_i);
        df_dx[i] = twoM * num / max(denom2, 1e-30);
    }

    // --- l vector (spatial part) and its Jacobian ---
    // l = (1, lx, ly, lz) where:
    //   lx = (r*px + a*pz) / (r^2 + a^2)
    //   ly = py / r
    //   lz = (r*pz - a*px) / (r^2 + a^2)
    float ra2 = r2 + a2;
    float inv_ra2 = 1.0 / max(ra2, 1e-10);
    float inv_r = 1.0 / max(r, 1e-10);

    vec3 lv = vec3(
    (r * p.x + a * p.z) * inv_ra2,
    p.y * inv_r,
    (r * p.z - a * p.x) * inv_ra2
    );

    // Jacobian of l: dlv_x[i] = d(lx)/d(x_i), etc.
    // Each component uses the quotient rule.
    float ra2_sq = ra2 * ra2;
    float inv_ra2_sq = 1.0 / max(ra2_sq, 1e-20);
    float inv_r2 = inv_r * inv_r;

    vec3 dlv_x, dlv_y, dlv_z;

    // d(lx)/d(x_i): lx = N_x / ra2, where N_x = r*px + a*pz
    float N_x = r * p.x + a * p.z;
    for (int i = 0; i < 3; i++) {
        float dr_i = dr_dx[i];
        // d(N_x)/d(x_i) = dr/dx_i * px + r * d(px)/d(x_i) + a * d(pz)/d(x_i)
        // d(px)/d(x_i) = 1 if i==0, else 0.  d(pz)/d(x_i) = 1 if i==2, else 0.
        float dN_x = dr_i * p.x;
        if (i == 0) dN_x += r;     // px depends on x
        if (i == 2) dN_x += a;     // pz depends on z, with coefficient a
        // quotient rule: d(N/D) = (dN*D - N*dD) / D^2
        // where dD = d(ra2)/dx_i = 2*r*dr/dx_i
        dlv_x[i] = (dN_x * ra2 - N_x * 2.0 * r * dr_i) * inv_ra2_sq;
    }

    // d(ly)/d(x_i): ly = py / r
    for (int i = 0; i < 3; i++) {
        float dr_i = dr_dx[i];
        // d(py)/d(x_i) = 1 if i==1, else 0
        float dN_y = (i == 1) ? 1.0 : 0.0;
        // quotient rule: d(py/r) = (d(py)*r - py*dr) / r^2
        dlv_y[i] = (dN_y * r - p.y * dr_i) * inv_r2;
    }

    // d(lz)/d(x_i): lz = N_z / ra2, where N_z = r*pz - a*px
    float N_z = r * p.z - a * p.x;
    for (int i = 0; i < 3; i++) {
        float dr_i = dr_dx[i];
        float dN_z = dr_i * p.z;
        if (i == 2) dN_z += r;     // pz depends on z
        if (i == 0) dN_z -= a;     // px depends on x, with coefficient -a
        dlv_z[i] = (dN_z * ra2 - N_z * 2.0 * r * dr_i) * inv_ra2_sq;
    }

    // --- Contract everything to get the acceleration ---
    //
    // Split the 4-velocity into time and spatial parts
    float vt = v4.x;
    vec3 vs = v4.yzw;

    // l dot v = l_t * vt + l_spatial dot v_spatial
    //         = 1*vt + lv dot vs  (since l_t = 1)
    float ldotv = vt + dot(lv, vs);

    // --- Q_beta: contraction of (velocity dotted into gradient of h) with velocity ---
    //
    // Since nothing depends on time, the velocity-directional-derivative
    // only involves spatial derivatives: v^i * d/dx_i
    //
    // Applied to h_{beta,nu} = f * l_beta * l_nu, this gives three pieces:
    //   (v dot grad f) * l_beta * l_nu
    //   + f * (v dot grad l_beta) * l_nu
    //   + f * l_beta * (v dot grad l_nu)
    //
    // Then contract with v^nu to get Q_beta.

    // v dot grad f (directional derivative of f along spatial velocity)
    float vdf = dot(vs, df_dx);

    // v dot grad l_j (directional derivative of each l component along vs)
    float vdl_x = dot(vs, dlv_x);
    float vdl_y = dot(vs, dlv_y);
    float vdl_z = dot(vs, dlv_z);
    vec3 vdl = vec3(vdl_x, vdl_y, vdl_z);

    // (v dot grad l_nu) contracted with v^nu
    // = (v dot grad l_t)*vt + (v dot grad l_spatial) dot vs
    // = 0 + vdl dot vs  (since l_t is constant, its gradient is zero)
    float vdl_dot_v = dot(vdl, vs);

    // Q_t (time component)
    float Qt_val = vdf * ldotv + f * vdl_dot_v;

    // Q_spatial (spatial components)
    // Q_j = l_j * Qt_val + f * vdl_j * ldotv
    vec3 Qs = lv * Qt_val + f * vdl * ldotv;

    // --- S_beta: half of (gradient of h) contracted with v twice ---
    //
    // For spatial index i:
    //   d_i(h_{u,v}) * v^u * v^v
    //   = (df/dx_i) * (l dot v)^2  +  2*f * (l dot v) * (dl_alpha/dx_i * v^alpha)
    //
    // The second factor (dl_alpha/dx_i * v^alpha) is NOT the same as vdl:
    //   vdl_j = v^i * dl_j/dx_i   (velocity contracts the FIRST index)
    //   here we need dl_alpha/dx_i * v^alpha  (velocity contracts the l index)
    //
    // So for each spatial direction i, we need:
    //   sum over alpha: (dl_alpha/dx_i) * v^alpha
    //   = dl_t/dx_i * vt + dl_x/dx_i * vx + dl_y/dx_i * vy + dl_z/dx_i * vz
    //   = 0 + dlv_x[i]*vs.x + dlv_y[i]*vs.y + dlv_z[i]*vs.z

    float ldv2 = ldotv * ldotv;

    vec3 dl_dot_v = vec3(
    dlv_x[0]*vs.x + dlv_y[0]*vs.y + dlv_z[0]*vs.z,   // for i=x
    dlv_x[1]*vs.x + dlv_y[1]*vs.y + dlv_z[1]*vs.z,   // for i=y
    dlv_x[2]*vs.x + dlv_y[2]*vs.y + dlv_z[2]*vs.z    // for i=z
    );

    vec3 Ss = 0.5 * (df_dx * ldv2 + 2.0 * f * ldotv * dl_dot_v);

    // S_t = 0 because nothing depends on time
    float St_val = 0.0;

    // --- Final acceleration ---
    //
    // Define R_beta = Q_beta - S_beta
    //
    // The inverse metric is g^{ab} = eta^{ab} - f * l^a * l^b
    // where l^a = (-1, lx, ly, lz) (time component flipped by eta)
    //
    // Acceleration:
    //   a^i = -R_i + f * l_i * (l^beta * R_beta)
    //
    // where l^beta * R_beta = -R_t + lv dot R_spatial

    float Rt = Qt_val - St_val;
    vec3 Rs = Qs - Ss;
    float lR = -Rt + dot(lv, Rs);

    vec3 acc = -Rs + f * lv * lR;

    return acc;
}


// ============================================================
// Helper: solve for vt from the null condition.
//
// The null condition g_{uv} v^u v^v = 0 expands to:
//   -vt^2 + |vs|^2 + f*(vt + lv dot vs)^2 = 0
//
// This is a quadratic in vt:
//   (f - 1)*vt^2 + 2*f*(lv dot vs)*vt + (|vs|^2 + f*(lv dot vs)^2) = 0
//
// We pick the root with vt > 0 (future-directed photon).
// For f -> 0 (flat space far from BH), vt -> 1 as expected.
// ============================================================
float solve_vt(vec3 p, vec3 vs) {
    float a2 = a * a;
    float rho2 = dot(p, p);
    float z_pole = p.y;
    float z2 = z_pole * z_pole;
    float w2 = rho2 - a2;
    float D = sqrt(max(w2 * w2 + 4.0 * a2 * z2, 0.0));
    float r2 = max(0.5 * (w2 + D), 1e-10);
    float r_val = sqrt(r2);

    float denom = r2 * r2 + a2 * z2;
    float f_val = 2.0 * M * r_val * r2 / max(denom, 1e-20);

    float ra2 = r2 + a2;
    vec3 lv_val = vec3(
    (r_val * p.x + a * p.z) / max(ra2, 1e-10),
    p.y / max(r_val, 1e-10),
    (r_val * p.z - a * p.x) / max(ra2, 1e-10)
    );

    float ldvs = dot(lv_val, vs);
    float A_q = f_val - 1.0;
    float B_q = 2.0 * f_val * ldvs;
    float C_q = dot(vs, vs) + f_val * ldvs * ldvs;
    float disc_q = max(B_q * B_q - 4.0 * A_q * C_q, 0.0);

    // The (-B - sqrt) root gives vt > 0 when f < 1 (outside ergosphere)
    return (-B_q - sqrt(disc_q)) / (2.0 * A_q);
}


void main() {
    // --- Build the camera ray direction ---
    // ndc goes from -1 to 1 across the screen
    // aspect ratio correction applied to horizontal axis
    vec2 ndc = uv * 2.0 - 1.0;
    float scale = tan(radians(fov) * 0.5);
    vec3 rayDir = normalize(camForward
    + ndc.x * scale * aspect * camRight
    + ndc.y * scale * camUp);

    // --- Initial conditions ---
    // Position is just the camera position in Cartesian
    vec3 pos = camPos;

    // Spatial velocity is the ray direction (will be rescaled by null condition)
    vec3 vs = rayDir;

    // Solve for the time component of the 4-velocity from the null condition
    float vt = solve_vt(pos, vs);
    vec4 vel = vec4(vt, vs);

    // Event horizon radius: r_+ = M + sqrt(M^2 - a^2)
    float a2_val = a * a;
    float horizon = M + sqrt(max(M * M - a2_val, 0.0));

    // --- Main ray marching loop (RK4 integration) ---
    for (int i = 0; i < MAX_STEPS; i++) {

        float r_cur = kerr_r(pos);

        // Check: did the ray fall into the black hole?
        if (r_cur < horizon * 1.01 + 0.01) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        // Check: did the ray escape to "infinity"?
        float d2 = dot(pos, pos);
        if (d2 > ESCAPE_R2) {
            // Look up the background texture using equirectangular mapping
            vec3 fd = normalize(pos);
            float u_tex = 0.5 + atan(fd.z, fd.x) / (2.0 * PI);
            float v_tex = 0.5 - asin(clamp(fd.y, -1.0, 1.0)) / PI;
            fragColor = texture(sceneTexture, vec2(u_tex, v_tex));
            return;
        }

        // Adaptive step size: small near the horizon, larger far away
        float dr_h = r_cur - horizon;
        float h = clamp(0.05 * dr_h, 0.005 * M, 0.5 * M);

        // --- RK4 integration of the second-order system ---
        // We're integrating:  d(pos)/dlambda = vs
        //                     d(vs)/dlambda  = acceleration
        // and recomputing vt at each substep from the null condition.

        // k1: acceleration at current state
        vec3 acc1 = geodesic_accel(pos, vel);
        vec3 k1_pos = vel.yzw;     // spatial velocity = d(pos)/dlambda
        vec3 k1_vel = acc1;         // acceleration = d(vs)/dlambda

        // k2: half step using k1
        vec3 p2 = pos + 0.5 * h * k1_pos;
        vec3 v2s = vs + 0.5 * h * k1_vel;
        float vt2 = solve_vt(p2, v2s);
        vec3 acc2 = geodesic_accel(p2, vec4(vt2, v2s));
        vec3 k2_pos = v2s;
        vec3 k2_vel = acc2;

        // k3: half step using k2
        vec3 p3 = pos + 0.5 * h * k2_pos;
        vec3 v3s = vs + 0.5 * h * k2_vel;
        float vt3 = solve_vt(p3, v3s);
        vec3 acc3 = geodesic_accel(p3, vec4(vt3, v3s));
        vec3 k3_pos = v3s;
        vec3 k3_vel = acc3;

        // k4: full step using k3
        vec3 p4 = pos + h * k3_pos;
        vec3 v4s = vs + h * k3_vel;
        float vt4 = solve_vt(p4, v4s);
        vec3 acc4 = geodesic_accel(p4, vec4(vt4, v4s));
        vec3 k4_pos = v4s;
        vec3 k4_vel = acc4;

        // Weighted RK4 update
        pos += h / 6.0 * (k1_pos + 2.0*k2_pos + 2.0*k3_pos + k4_pos);
        vs  += h / 6.0 * (k1_vel + 2.0*k2_vel + 2.0*k3_vel + k4_vel);

        // Recompute vt at the new position to stay on the null geodesic
        vt = solve_vt(pos, vs);
        vel = vec4(vt, vs);
    }

    // Ray didn't hit anything or escape -- treat as black
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}