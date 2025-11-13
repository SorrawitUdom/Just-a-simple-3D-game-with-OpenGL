#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iostream>
#include <vector>
#include <algorithm>

// --------- Tunables ---------
#define CAR_SPEED 3.5f
#define CAR_SPEED_R 2.5f
#define CAR_SPEED_BOOST_FACTOR 3.0f
#define ROTATION_SPEED 0.01f

// screen
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// car transform state
glm::vec3 model_trans_loc = glm::vec3(0.0f, 0.0f, 0.0f);
float rotation = 0.0f;

// camera
Camera camera(glm::vec3(0.0f, 1.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// forward decl
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// --------- Collision types & helpers ---------
struct AABB {
    glm::vec3 minLocal; // using "Local" name from your original — these are just mins/maxes
    glm::vec3 maxLocal;
};

typedef struct building_t{
    Model *buildingModel;
    glm::vec3 buildingPos;
    glm::vec3 buildingScaleFactor; // non-uniform supported
    float buildingRotation;        // ignored by AABB system (use OBB for rotated)
} BUILDING_T;

// Local-space AABBs (estimate & tweak for your meshes):
// Car mesh in modeled units — adjust after a quick visual test.
static const AABB kCarLocalAABB = {
    glm::vec3(-0.9f, 0.0f, -1.9f), // min
    glm::vec3( 0.9f, 1.5f,  1.9f)  // max
};

// Building mesh base AABB (authoring units, before scene scale)
// Adjust once to your building.obj bounds.
static const AABB kBuildingLocalAABB = {
    glm::vec3(-10.0f, 0.0f, -8.0f),
    glm::vec3( 10.0f,10.0f,  8.0f)
};

// All buildings live here
std::vector<BUILDING_T> gBuildings;

// last safe car position
glm::vec3 moment_before_collision;

// Build a world-space AABB from a local AABB, given pos & non-uniform scale
inline void toWorldAABB_NonRotated(const AABB& localBox, const glm::vec3& pos, const glm::vec3& scale, AABB& outWorld)
{
    // scale each corner, then compute min/max (handles non-uniform scale)
    glm::vec3 c0 = localBox.minLocal * scale;
    glm::vec3 c1 = glm::vec3(localBox.minLocal.x, localBox.minLocal.y, localBox.maxLocal.z) * scale;
    glm::vec3 c2 = glm::vec3(localBox.minLocal.x, localBox.maxLocal.y, localBox.minLocal.z) * scale;
    glm::vec3 c3 = glm::vec3(localBox.minLocal.x, localBox.maxLocal.y, localBox.maxLocal.z) * scale;
    glm::vec3 c4 = glm::vec3(localBox.maxLocal.x, localBox.minLocal.y, localBox.minLocal.z) * scale;
    glm::vec3 c5 = glm::vec3(localBox.maxLocal.x, localBox.minLocal.y, localBox.maxLocal.z) * scale;
    glm::vec3 c6 = glm::vec3(localBox.maxLocal.x, localBox.maxLocal.y, localBox.minLocal.z) * scale;
    glm::vec3 c7 = localBox.maxLocal * scale;

    glm::vec3 mn = glm::min(glm::min(glm::min(c0,c1), glm::min(c2,c3)), glm::min(glm::min(c4,c5), glm::min(c6,c7)));
    glm::vec3 mx = glm::max(glm::max(glm::max(c0,c1), glm::max(c2,c3)), glm::max(glm::max(c4,c5), glm::max(c6,c7)));

    outWorld.minLocal = mn + pos;
    outWorld.maxLocal = mx + pos;
}

// Overlap test for two world AABBs
inline bool aabbOverlap(const AABB& a, const AABB& b)
{
    if (a.maxLocal.x < b.minLocal.x || a.minLocal.x > b.maxLocal.x) return false;
    if (a.maxLocal.y < b.minLocal.y || a.minLocal.y > b.maxLocal.y) return false;
    if (a.maxLocal.z < b.minLocal.z || a.minLocal.z > b.maxLocal.z) return false;
    return true;
}

// Rotate a point p around Y by yaw (in radians)
inline glm::vec3 rotateY(const glm::vec3& p, float yaw)
{
    float c = cosf(yaw);
    float s = sinf(yaw);
    return glm::vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);
}

// Build the car's *rotation-aware* world AABB at position `carPos` and yaw `carYaw`.
// We rotate the 8 local corners around Y, then translate by carPos, and take min/max.
inline void carWorldAABBAt(const glm::vec3& carPos, float carYaw, AABB& outWorld)
{
    // car scale = 1, so just rotate local corners
    glm::vec3 mn = kCarLocalAABB.minLocal;
    glm::vec3 mx = kCarLocalAABB.maxLocal;

    glm::vec3 corners[8] = {
        {mn.x, mn.y, mn.z},
        {mn.x, mn.y, mx.z},
        {mn.x, mx.y, mn.z},
        {mn.x, mx.y, mx.z},
        {mx.x, mn.y, mn.z},
        {mx.x, mn.y, mx.z},
        {mx.x, mx.y, mn.z},
        {mx.x, mx.y, mx.z}
    };

    glm::vec3 r0 = rotateY(corners[0], carYaw) + carPos;
    glm::vec3 r1 = rotateY(corners[1], carYaw) + carPos;
    glm::vec3 r2 = rotateY(corners[2], carYaw) + carPos;
    glm::vec3 r3 = rotateY(corners[3], carYaw) + carPos;
    glm::vec3 r4 = rotateY(corners[4], carYaw) + carPos;
    glm::vec3 r5 = rotateY(corners[5], carYaw) + carPos;
    glm::vec3 r6 = rotateY(corners[6], carYaw) + carPos;
    glm::vec3 r7 = rotateY(corners[7], carYaw) + carPos;

    glm::vec3 outMin = glm::min(glm::min(glm::min(r0,r1), glm::min(r2,r3)),
                                glm::min(glm::min(r4,r5), glm::min(r6,r7)));
    glm::vec3 outMax = glm::max(glm::max(glm::max(r0,r1), glm::max(r2,r3)),
                                glm::max(glm::max(r4,r5), glm::max(r6,r7)));

    outWorld.minLocal = outMin;
    outWorld.maxLocal = outMax;
}

// Does the car at `proposedCarPos` with yaw `proposedYaw` overlap any building AABB?
bool wouldCollideAt(const glm::vec3& proposedCarPos, float proposedYaw)
{
    AABB carW;
    carWorldAABBAt(proposedCarPos, proposedYaw, carW);

    for (const auto& b : gBuildings) {
        AABB bW;
        toWorldAABB_NonRotated(kBuildingLocalAABB, b.buildingPos, b.buildingScaleFactor, bW);
        if (aabbOverlap(carW, bW)) return true;
    }
    return false;
}

// --------- Rendering helpers ---------
void drawBuilding(BUILDING_T *building ,Shader shader){
    glm::mat4 buildingModel = glm::mat4(1.0f);
    buildingModel = glm::translate(buildingModel, building->buildingPos);
    buildingModel = glm::rotate(buildingModel, building->buildingRotation, glm::vec3(0.0f, 1.0f, 0.0f)); // ignored by AABB
    buildingModel = glm::scale(buildingModel, building->buildingScaleFactor);
    shader.setMat4("model", buildingModel);
    building->buildingModel->Draw(shader);
}

// --------- Main ---------
int main()
{
    // glfw init
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "My Game (AABB collisions w/ rotation check)", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // capture mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // images upright
    stbi_set_flip_vertically_on_load(false);

    // GL state
    glEnable(GL_DEPTH_TEST);

    // shaders
    Shader ourShader("1.model_loading.vs", "1.model_loading.fs");

    // load models
    Model carModel(FileSystem::getPath("resources/assignment_3/obj/exported_car/car.obj"));
    Model *buildingModelPtr = new Model(FileSystem::getPath("resources/assignment_3/obj/exported_building/building.obj"));

    // add buildings (add as many as you like)
    gBuildings.push_back({ buildingModelPtr, glm::vec3( 0.0f, 0.0f, -5.0f), glm::vec3(0.04f, 0.04f, 0.04f), glm::radians(180.0f) });
    gBuildings.push_back({ buildingModelPtr, glm::vec3( 8.0f, 0.0f,-12.0f), glm::vec3(0.05f, 0.05f, 0.05f), 0.0f });
    gBuildings.push_back({ buildingModelPtr, glm::vec3(-6.0f, 0.0f,  2.0f), glm::vec3(0.035f,0.035f,0.035f), 0.0f });

    moment_before_collision = model_trans_loc;


    Shader FloorShader("7.4.camera.vs", "7.4.camera.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);


    // load and create a texture 
    // -------------------------
    unsigned int texture1, texture2;
    // texture 1
    // ---------
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    unsigned char *data = stbi_load(FileSystem::getPath("resources/textures/container.jpg").c_str(), &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    // texture 2
    // ---------
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    data = stbi_load(FileSystem::getPath("resources/textures/grass.jpg").c_str(), &width, &height, &nrChannels, 0);
    if (data)
    {
        // note that the awesomeface.png has transparency and thus an alpha channel, so make sure to tell OpenGL the data type is of GL_RGBA
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    // -------------------------------------------------------------------------------------------
    FloorShader.use();
    FloorShader.setInt("texture1", 0);
    FloorShader.setInt("texture2", 1);




    // render loop
    while (!glfwWindowShouldClose(window))
    {
        // time
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        processInput(window);

        // clear
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();

        glm::mat4 model;

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        // activate shader
        FloorShader.use();

        // pass projection matrix to shader (note that in this case it could change every frame)
        FloorShader.setMat4("projection", projection);

        // camera/view transformation

        FloorShader.setMat4("view", view);

        // render boxes
        glBindVertexArray(VAO);
        // calculate the model matrix for each object and pass it to shader before drawing
        model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        model = glm::translate(model, glm::vec3(0.0f,-2.0f,0.0f));
        model = glm::scale(model,glm::vec3(100.0f,1.0f,100.0f));
        FloorShader.setMat4("model", model);

        glDrawArrays(GL_TRIANGLES, 0, 36);







        // shader
        ourShader.use();

        // car model matrix
        model = glm::mat4(1.0f);
        model = glm::translate(model, model_trans_loc);
        model = glm::rotate(model, rotation, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(1.0f));
        ourShader.setMat4("model", model);
        carModel.Draw(ourShader);

        // buildings
        for (auto& b : gBuildings) {
            drawBuilding(&b, ourShader);
        }

        // third-person-ish chase camera
        glm::vec3 ourPos = glm::vec3(model_trans_loc.x, model_trans_loc.y + 8.0f, model_trans_loc.z - 3.0f);
        camera.Yaw = -270.0f + rotation;
        camera.Pitch = -60.0f;
        camera.Position = ourPos;

        // view/projection

        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);

        // swap/poll
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    delete buildingModelPtr;
    return 0;
}

// --------- Input with rotation-gated collision + axis-wise sliding ---------
bool wouldCollideAt(const glm::vec3& proposedCarPos, float proposedYaw); // fwd (already defined above, but some compilers like this here)

// Propose movement OR rotation first, test, then commit.
// Rotation is blocked if it would cause an overlap at the *current* position.
void processInput(GLFWwindow *window)
{
    float speed = CAR_SPEED;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS && glfwGetKey(window, GLFW_KEY_S) != GLFW_PRESS){
        speed *= CAR_SPEED_BOOST_FACTOR;
    }

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // --- Rotation: propose → test → commit ---
    float proposedYaw = rotation;
    bool rotated = false;

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        proposedYaw += ROTATION_SPEED;
        rotated = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        proposedYaw -= ROTATION_SPEED;
        rotated = true;
    }

    // clamp to [-pi, pi] keeps numbers tame (optional)
    auto wrapPi = [](float a)->float {
        if (a >  glm::pi<float>()) a -= 2.0f * glm::pi<float>();
        if (a < -glm::pi<float>()) a += 2.0f * glm::pi<float>();
        return a;
    };
    proposedYaw = wrapPi(proposedYaw);

    if (rotated) {
        // only accept the rotation if it doesn't create a collision at the current position
        if (!wouldCollideAt(model_trans_loc, proposedYaw)) {
            rotation = proposedYaw;
        }
        // else: rotation blocked; keep old rotation
    }

    // --- Translation: propose → test → commit (uses current rotation) ---
    glm::vec3 proposedPos = model_trans_loc;
    float fwd = 0.0f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) fwd += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) fwd -= 1.0f;

    if (fwd != 0.0f) {
        float step = fwd * speed * deltaTime;
        proposedPos.z += step * cos(rotation);
        proposedPos.x += step * sin(rotation);
    }

    if (!wouldCollideAt(proposedPos, rotation)) {
        model_trans_loc = proposedPos;
        moment_before_collision = model_trans_loc;
    } else {
        // try sliding along axes (still using current rotation)
        glm::vec3 slideX = glm::vec3(proposedPos.x, model_trans_loc.y, model_trans_loc.z);
        glm::vec3 slideZ = glm::vec3(model_trans_loc.x, model_trans_loc.y, proposedPos.z);

        bool xFree = !wouldCollideAt(slideX, rotation);
        bool zFree = !wouldCollideAt(slideZ, rotation);

        if (xFree && !zFree) {
            model_trans_loc.x = slideX.x;
            moment_before_collision = model_trans_loc;
        } else if (!xFree && zFree) {
            model_trans_loc.z = slideZ.z;
            moment_before_collision = model_trans_loc;
        } else {
            // blocked both ways: stay put at last safe
            model_trans_loc = moment_before_collision;
        }
    }
}

// --------- GLFW callbacks ---------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    (void)xoffset;
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
