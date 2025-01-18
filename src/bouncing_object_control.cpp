#include <ros/ros.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/GetModelState.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/UInt8.h>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// Global variables
gazebo_msgs::ModelState sphere_state;
gazebo_msgs::ModelState bouncing_state;
ros::ServiceClient set_state_client;
ros::ServiceClient get_state_client;
ros::Publisher cable_length_pub;
ros::Publisher motor_command_pub;
std_msgs::Int16MultiArray motor_commands;
geometry_msgs::Vector3 mapped_position;
bool is_moving = false;
bool is_reset = true;  // Start in reset state
bool is_mapping = false;
double x_scale = 1.0;
double x_offset = 0.0;
double z_scale = 1.0;
double z_offset = 0.0;
geometry_msgs::Vector3 map_point;

geometry_msgs::Vector3 current_hand_position;
bool is_hand_controlled = true;
bool is_waiting_for_launch = true;
const double RESET_SPEED = 0.1;  
const double CATCH_DISTANCE = 2.0;
const double LAUNCH_SPEED = 0.2;

// Add to global variables
struct CalibrationPoints {
    double x_min;  // For key 4
    double x_max;  // For key 6
    double z_min;  // For key 2
    double z_max;  // For key 8
    bool x_calibrated;
    bool z_calibrated;
} calib_points = {0, 0, 0, 0, false, false};

// Initial position constants
const double INITIAL_X = 0.0;
const double INITIAL_Y = 8.0;
const double INITIAL_Z = 4.0;

// Sound settings
const std::string SOUND_BASE_PATH = "/home/shoes-pc/catkin_ws/src/hand_tracking/sound/";
const std::string WALL_SOUND = SOUND_BASE_PATH + "wall.mp3";
const std::string GOAL_SOUND = SOUND_BASE_PATH + "goal.mp3";
ros::Time last_sound_time;
const double SOUND_COOLDOWN = 0.1;

// Bouncing object parameters
double velocity_x = 0.0;
double velocity_y = 0.0;
double velocity_z = 0.0;

ros::Time last_motor_command_time;
const double MOTOR_COMMAND_PERIOD = 0.1;

// Sphere control
std::map<uint8_t, std::string> sphere_names = {
    {255, "black_sphere"},
    {0, "blue_sphere"},
    {100, "red_sphere"},
    {200, "green_sphere"}
};
std::string current_sphere = "black_sphere";
gazebo_msgs::ModelState sphere_states[4];

// Fixed points
std::vector<std::string> fixed_point_names = {
    "fixed_point1", "fixed_point2", "fixed_point3", "fixed_point4",
    "fixed_point5", "fixed_point6", "fixed_point7", "fixed_point8"
};

// Update mapping function
geometry_msgs::Vector3 mapHandPosition(const geometry_msgs::Vector3& raw_pos) {
    geometry_msgs::Vector3 mapped_pos;
    
    // X mapping using calibration points
    mapped_pos.x = -4.0 + (raw_pos.x - calib_points.x_min) * 8.0 / 
                      (calib_points.x_max - calib_points.x_min);


    // Y remains unchanged
    mapped_pos.y = raw_pos.y;
    
    // Z mapping using calibration points
    mapped_pos.z = (raw_pos.z - calib_points.z_min) * 8.0 / 
                      (calib_points.z_max - calib_points.z_min);

    
    return mapped_pos;
}

int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    if(ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

void playSound(const std::string& sound_file) {
    if ((ros::Time::now() - last_sound_time).toSec() > SOUND_COOLDOWN) {
        std::string cmd = "play " + sound_file + " 2>/dev/null &";
        system(cmd.c_str());
        last_sound_time = ros::Time::now();
    }
}

bool isInGoalRegion(const geometry_msgs::Pose& pose) {
    return (pose.position.x > -2.0 && pose.position.x < 2.0 &&
            pose.position.y > 0.0 && pose.position.y < 1.0 &&
            pose.position.z > 2.0 && pose.position.z < 6.0);
}

// Update resetBouncingObject function
void resetBouncingObject() {
    // Calculate direction vector to initial position
    double dx = INITIAL_X - bouncing_state.pose.position.x;
    double dy = INITIAL_Y - bouncing_state.pose.position.y;
    double dz = INITIAL_Z - bouncing_state.pose.position.z;
    
    // Normalize direction vector
    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (distance > 0.01) {  // Still moving
        velocity_x = (dx / distance) * RESET_SPEED;
        velocity_y = (dy / distance) * RESET_SPEED;
        velocity_z = (dz / distance) * RESET_SPEED;
        is_reset = true;
        is_moving = false;
    } else {  // Reached target
        bouncing_state.pose.position.x = INITIAL_X;
        bouncing_state.pose.position.y = INITIAL_Y;
        bouncing_state.pose.position.z = INITIAL_Z;
        velocity_x = 0.0;
        velocity_y = 0.0;
        velocity_z = 0.0;
        is_reset = true;
        is_moving = false;
    }
}

void updateSpherePosition(const geometry_msgs::Vector3& pos) {
    for (auto& state : sphere_states) {
        if (state.model_name == current_sphere) {
            state.pose.position.x = pos.x;
            state.pose.position.y = pos.y;
            state.pose.position.z = pos.z;
            
            gazebo_msgs::SetModelState srv;
            srv.request.model_state = state;
            if (!set_state_client.call(srv)) {
                ROS_ERROR("Failed to set sphere state for %s", current_sphere.c_str());
            }
            break;
        }
    }
}

// Update position callback
void positionCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    current_hand_position = *msg;
    mapped_position = mapHandPosition(*msg);
    
    if (is_hand_controlled) {
        bouncing_state.pose.position.x = mapped_position.x;
        bouncing_state.pose.position.y = mapped_position.y;
        bouncing_state.pose.position.z = mapped_position.z;
        
        static geometry_msgs::Vector3 last_position = mapped_position;
        velocity_x = (mapped_position.x - last_position.x) * 30.0;
        velocity_y = (mapped_position.y - last_position.y) * 30.0;
        velocity_z = (mapped_position.z - last_position.z) * 30.0;
        last_position = mapped_position;
        
        gazebo_msgs::SetModelState srv_bounce;
        srv_bounce.request.model_state = bouncing_state;
        set_state_client.call(srv_bounce);
    }
    updateSpherePosition(mapped_position);
}

void gestureCallback(const std_msgs::UInt8::ConstPtr& msg) {
    uint8_t gesture = msg->data;
    current_sphere = sphere_names[gesture];
    
    if (gesture == 200) {  // PEACE
        // Calculate distance between hand and bouncing object
        double dx = mapped_position.x - bouncing_state.pose.position.x;
        double dy = mapped_position.y - bouncing_state.pose.position.y;
        double dz = mapped_position.z - bouncing_state.pose.position.z;
        double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (distance < CATCH_DISTANCE && 
            mapped_position.x > -4.0 && mapped_position.x < 4.0 &&
            mapped_position.y > 0.0 && mapped_position.y < 4.0 &&
            mapped_position.z > 0.0 && mapped_position.z < 8.0) {
            is_hand_controlled = true;
            is_moving = false;
            is_reset = false;
            // Set bouncing object to hand position
            bouncing_state.pose.position.x = mapped_position.x;
            bouncing_state.pose.position.y = mapped_position.y;
        }
    }
    else if (gesture == 0 && is_hand_controlled) {  // OPEN
        is_waiting_for_launch = true;
        is_hand_controlled = false;
    }
    else if (gesture == 100 && is_waiting_for_launch) {  // CLOSED
        double dx = mapped_position.x - bouncing_state.pose.position.x;
        double dy = mapped_position.y - bouncing_state.pose.position.y;
        double dz = mapped_position.z - bouncing_state.pose.position.z;
        double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance < CATCH_DISTANCE) {
            is_waiting_for_launch = false;
            is_moving = true;
            double total_velocity = std::sqrt(velocity_x*velocity_x + 
                                            velocity_y*velocity_y + 
                                            velocity_z*velocity_z);
            if (total_velocity > 0) {
                velocity_x = (velocity_x / total_velocity) * LAUNCH_SPEED;
                velocity_y = (velocity_y / total_velocity) * LAUNCH_SPEED;
                velocity_z = (velocity_z / total_velocity) * LAUNCH_SPEED;
            } else {
                velocity_x = LAUNCH_SPEED;
                velocity_y = 0;
                velocity_z = 0;
            }
        }
    }
}

void calculateAndPublishCableLengths() {
    
    std_msgs::Float64MultiArray cable_lengths;
    motor_commands.data.clear();
    cable_lengths.data.clear();
    std::stringstream ss;
    ss << "Cable lengths: [";

    for (const auto& point_name : fixed_point_names) {
        gazebo_msgs::GetModelState get_state;
        get_state.request.model_name = point_name;
        get_state.request.relative_entity_name = "world";

        if (get_state_client.call(get_state)) {
            // Create offset point based on fixed point name
            double offset_x = 0.0, offset_y = 0.0;
            if (point_name == "fixed_point1" || point_name == "fixed_point5") {
                offset_x = 0.45 * std::sqrt(2);
                offset_y = 0.45 * std::sqrt(2);
            } else if (point_name == "fixed_point2" || point_name == "fixed_point6") {
                offset_x = -0.45 * std::sqrt(2);
                offset_y = 0.45 * std::sqrt(2);
            } else if (point_name == "fixed_point3" || point_name == "fixed_point7") {
                offset_x = -0.45 * std::sqrt(2);
                offset_y = -0.45 * std::sqrt(2);
            } else if (point_name == "fixed_point4" || point_name == "fixed_point8") {
                offset_x = 0.45 * std::sqrt(2);
                offset_y = -0.45 * std::sqrt(2);
            }

            // Calculate length considering offsets and bouncing object radius
            double length = std::sqrt(
                std::pow(get_state.response.pose.position.x - 
                        (bouncing_state.pose.position.x + offset_x), 2) +
                std::pow(get_state.response.pose.position.y - 
                        (bouncing_state.pose.position.y + offset_y), 2) +
                std::pow(get_state.response.pose.position.z - 
                        bouncing_state.pose.position.z, 2)
            );

            cable_lengths.data.push_back(length);
            motor_commands.data.push_back(static_cast<int>((length-7.8) * 120));
            ss << length << ", ";
        } else {
            ROS_ERROR("Failed to get state for %s", point_name.c_str());
            cable_lengths.data.push_back(0.0);
            motor_commands.data.push_back(0);
            ss << "0.0, ";
        }
    }
    ss << "]";

    cable_length_pub.publish(cable_lengths);
    motor_command_pub.publish(motor_commands);
    last_motor_command_time = ros::Time::now();
    ROS_INFO_THROTTLE(1.0, "%s", ss.str().c_str());
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "bouncing_object_control");
    ros::NodeHandle nh;
    
    set_state_client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
    get_state_client = nh.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    cable_length_pub = nh.advertise<std_msgs::Float64MultiArray>("cable_lengths", 10);
    motor_command_pub = nh.advertise<std_msgs::Int16MultiArray>("motor_commands", 10);

    ros::Subscriber position_sub = nh.subscribe("/hand_tracking/position", 10, positionCallback);
    ros::Subscriber gesture_sub = nh.subscribe("/hand_tracking/gesture", 10, gestureCallback);

    for (int i = 0; i < 4; ++i) {
        sphere_states[i].model_name = sphere_names[i == 0 ? 255 : i == 1 ? 0 : i == 2 ? 100 : 200];
        sphere_states[i].pose.orientation.w = 1.0;
    }

    bouncing_state.model_name = "bouncing_object";
    resetBouncingObject();
    bouncing_state.pose.orientation.w = 1.0;

    double angle = M_PI / 6;
    double speed = 0.1;

    is_hand_controlled = false;
    is_waiting_for_launch = false;

    ros::Rate rate(30);
    while (ros::ok()) {
        if (kbhit()) {
            char c = getchar();
            if ((c == 's' || c == 'S')) {
                is_moving = true;
                is_reset = false;
                velocity_x = speed * cos(angle);
                velocity_y = speed * sin(angle);
                velocity_z = -speed * sin(angle);
            }
            if (c == 't' || c == 'T') {
                is_moving = false;
                resetBouncingObject();
            }
            if (c == '4') {
                calib_points.x_min = current_hand_position.x;
            }
            else if (c == '6') {
                calib_points.x_max = current_hand_position.x;
            }
            else if (c == '2') {
                calib_points.z_min = current_hand_position.z;
            }
            else if (c == '8') {
                calib_points.z_max = current_hand_position.z;
            }
        }

        if (is_moving) {
            bouncing_state.pose.position.x += velocity_x;
            bouncing_state.pose.position.y += velocity_y;
            bouncing_state.pose.position.z += velocity_z;

            static bool wasInGoalRegion = false;
            bool isInGoal = isInGoalRegion(bouncing_state.pose);
            if (isInGoal && !wasInGoalRegion) {
                playSound(GOAL_SOUND);
            }
            wasInGoalRegion = isInGoal;

            if (abs(bouncing_state.pose.position.x) > 3.4) {
                velocity_x = -velocity_x;
                bouncing_state.pose.position.x = std::copysign(3.4, bouncing_state.pose.position.x);
                playSound(WALL_SOUND);
            }
            if (bouncing_state.pose.position.y > 15.0 || bouncing_state.pose.position.y < 1.0) {
                velocity_y = -velocity_y;
                bouncing_state.pose.position.y = 
                    bouncing_state.pose.position.y > 15.0 ? 15.0 : 1.0;
                playSound(WALL_SOUND);
            }
            if (bouncing_state.pose.position.z > 7.0 || bouncing_state.pose.position.z < 1.0) {
                velocity_z = -velocity_z;
                bouncing_state.pose.position.z = 
                    bouncing_state.pose.position.z > 7.0 ? 7.0 : 1.0;
                playSound(WALL_SOUND);
            }

            gazebo_msgs::SetModelState srv_bounce;
            srv_bounce.request.model_state = bouncing_state;
            if (!set_state_client.call(srv_bounce)) {
                ROS_ERROR("Failed to set bouncing object state");
            }
        } else if (is_reset) {
            bouncing_state.pose.position.x += velocity_x;
            bouncing_state.pose.position.y += velocity_y;
            bouncing_state.pose.position.z += velocity_z;
            resetBouncingObject();  // Continue checking for completion
            
            gazebo_msgs::SetModelState srv_bounce;
            srv_bounce.request.model_state = bouncing_state;
            set_state_client.call(srv_bounce);

        }else if (is_hand_controlled || is_waiting_for_launch) {
            gazebo_msgs::SetModelState srv_bounce;
            srv_bounce.request.model_state = bouncing_state;
            if (!set_state_client.call(srv_bounce)) {
                ROS_ERROR("Failed to set bouncing object state");
            }
        }

        calculateAndPublishCableLengths();
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}