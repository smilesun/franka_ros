// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/joint_impedance_example_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

namespace franka_example_controllers {

bool JointImpedanceExampleController::init(hardware_interface::RobotHW* robot_hw,
                                           ros::NodeHandle& node_handle) {
  std::string arm_id;
  //In this context, arm_id is the string identifier for the Franka arm (e.g., panda). It’s used to
  //build ROS interface/resource names like
  //  "<arm_id>_joint1" or "/<arm_id>_controller", so the controller talks to the correct robot
  //  instance. It’s typically read from a ROS
  //    parameter in the controller’s init().
  if (!node_handle.getParam("arm_id", arm_id)) {  // ROS read from config file
    ROS_ERROR("JointImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }

  // radius is from ROS config, used to draw circle
  if (!node_handle.getParam("radius", radius_)) {
    ROS_INFO_STREAM(
        "JointImpedanceExampleController: No parameter radius, defaulting to: " << radius_);
  }
  if (std::fabs(radius_) < 0.005) {
    ROS_INFO_STREAM("JointImpedanceExampleController: Set radius to small, defaulting to: " << 0.1);
    radius_ = 0.1;
  }

  if (!node_handle.getParam("vel_max", vel_max_)) {
    ROS_INFO_STREAM(
        "JointImpedanceExampleController: No parameter vel_max, defaulting to: " << vel_max_);
  }
  if (!node_handle.getParam("acceleration_time", acceleration_time_)) {
    ROS_INFO_STREAM(
        "JointImpedanceExampleController: No parameter acceleration_time, defaulting to: "
        << acceleration_time_);
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "JointImpedanceExampleController: Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("k_gains", k_gains_) || k_gains_.size() != 7) {
    ROS_ERROR(
        "JointImpedanceExampleController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("d_gains", d_gains_) || d_gains_.size() != 7) {
    ROS_ERROR(
        "JointImpedanceExampleController:  Invalid or no d_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  double publish_rate(30.0);
  if (!node_handle.getParam("publish_rate", publish_rate)) {
    ROS_INFO_STREAM("JointImpedanceExampleController: publish_rate not found. Defaulting to "
                    << publish_rate);
  }
  rate_trigger_ = franka_hw::TriggerRate(publish_rate);  // Throttle(limit) periodic publishing to this 
  // rate, instead of every control cycle
  // coriolis_factor is set to be 1 in header files
  if (!node_handle.getParam("coriolis_factor", coriolis_factor_)) {
    ROS_INFO_STREAM("JointImpedanceExampleController: coriolis_factor not found. Defaulting to "
                    << coriolis_factor_);
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointImpedanceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  } // catch

  auto* cartesian_pose_interface = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointImpedanceExampleController: Error getting cartesian pose interface from hardware");
    return false;
  }
  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointImpedanceExampleController: Exception getting cartesian pose handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>(); 
  // EffortJointInterface defines the handle is for torque control

  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointImpedanceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      // std::vector<hardware_interface::JointHandle> joint_handles_;
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointImpedanceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }
  torques_publisher_.init(node_handle, "torque_comparison", 1);

  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0);

  return true;
}

void JointImpedanceExampleController::starting(const ros::Time& /*time*/) {
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
}

void JointImpedanceExampleController::update(const ros::Time& /*time*/,  // Control loop update
                                             const ros::Duration& period) {  // Time since last update
  // tau_d = coriolis_factor * coriolis + K * (q_d - q) + D * (dq_d - dq_filtered)
  // - q_d, dq_d are the desired joint position/velocity from the robot’s trajectory generator.
  // - q, dq_filtered are the measured joint position/velocity.
  // - K is stiffness (spring), D is damping (damper).
  // - Coriolis compensation helps cancel dynamics.
  if (vel_current_ < vel_max_) {  // Ramp up velocity until the configured max
    vel_current_ += period.toSec() * std::fabs(vel_max_ / acceleration_time_);  // Acceleration step
  }  // End velocity ramp check
  vel_current_ = std::fmin(vel_current_, vel_max_);  // Clamp velocity to maximum
  // The robot’s internal trajectory generator produces q_d/dq_d for that pose, and the impedance
  //  law uses those in the torque computation.
  // The above line set how fast the desired motion evolves, which changes the target the impedance
  // controller is “spring‑damping” toward.

  angle_ += period.toSec() * vel_current_ / std::fabs(radius_);  // Advance trajectory phase
  if (angle_ > 2 * M_PI) {  // Wrap angle after full revolution
    angle_ -= 2 * M_PI;  // Keep angle within [0, 2*pi]
  }  // End angle wrap

  double delta_y = radius_ * (1 - std::cos(angle_));  // Y offset along circular path
  double delta_z = radius_ * std::sin(angle_);  // Z offset along circular path

  std::array<double, 16> pose_desired = initial_pose_;  // Start from initial pose
  pose_desired[13] += delta_y;  // Apply Y translation in pose matrix
  pose_desired[14] += delta_z;  // Apply Z translation in pose matrix
  // pose_desired is a 4×4 homogeneous transform matrix flattened into a 16‑element array (the
  // Franka API uses column‑major order). It
  //   represents the desired end‑effector pose.
  //
  //     In that layout:
  //
  //       index:  0  4  8 12
  //               1  5  9 13
  //               2  6 10 14
  //               3  7 11 15
  //
  //              So:
  //
  //              - pose_desired[12], [13], [14] are the x, y, z translation components.
  //              - pose_desired[13] specifically is the Y translation.
  //
  cartesian_pose_handle_->setCommand(pose_desired);  // Send desired Cartesian pose

  franka::RobotState robot_state = cartesian_pose_handle_->getRobotState();  // Read robot state
  std::array<double, 7> coriolis = model_handle_->getCoriolis();  // Get Coriolis torques
  std::array<double, 7> gravity = model_handle_->getGravity();  // Get gravity torques

  double alpha = 0.99;  // Low-pass filter coefficient for joint velocities
  for (size_t i = 0; i < 7; i++) {  // Loop over all joints
    dq_filtered_[i] = (1 - alpha) * dq_filtered_[i] + alpha * robot_state.dq[i];  // Filter dq
  }  // End velocity filter loop

  std::array<double, 7> tau_d_calculated;  // Desired torque before rate limiting
  for (size_t i = 0; i < 7; ++i) {  // Loop over all joints
    tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +  // Coriolis compensation
                          k_gains_[i] * (robot_state.q_d[i] - robot_state.q[i]) +  // P term
                          d_gains_[i] * (robot_state.dq_d[i] - dq_filtered_[i]);  // D term
    // "desired" - "measured"
  }  // End torque computation loop

  // Bad design from franka: the desired states is stored in robot_state as well, which is
  // automatically populated:
  //
  //  - Simulation (gazebo): franka_gazebo/src/franka_hw_sim.cpp lines 666–668:
  //      ```
  //      robot_state_.q_d[i]  = joint->getDesiredPosition(mode);
  //      robot_state_.dq_d[i] = joint->getDesiredVelocity(mode);
  //      robot_state_.ddq_d[i] = joint->getDesiredAcceleration(mode);
  //      ```
  //    This is where the “desired” joint state is written in sim.
  //
  //  - Real robot: franka_hw/src/franka_combinable_hw.cpp line 75:
  //      ```
  //      robot_state_libfranka_ = robot_->readOnce();
  //      ```
  //    returns a franka::RobotState from libfranka that already contains q_d. That state is then copied
  //    into robot_state_ros_ in franka_hw/src/franka_hw.cpp line 385.
  //
  //    So robot_state.q_d is populated by the driver (libfranka or gazebo sim), and your controller 
  //    just reads it.
  //
  // customization:
  // Compute your own q_des and dq_des in the controller and use those in the impedance law instead
  // of robot_state.q_d/dq_d. 
  // Example (inside update()):
  //
  //           std::array<double, 7> q_des = myTrajectory(...);
  //           std::array<double, 7> dq_des = myTrajectoryVel(...);
  //
//             tau_d_calculated[i] = coriolis_factor_ * coriolis[i] +
//                                                k_gains_[i] * (q_des[i] - robot_state.q[i]) +
//                                                                           d_gains_[i]
//                                                                           * (dq_des[i]
//                                                                           - dq_filtered_[i]);
// to have dynamic desired, use
// double t = elapsed_time_.toSec();
//   for (size_t i = 0; i < 7; ++i) {
//       q_des[i]  = q_start[i] + A[i] * std::sin(omega * t);
//           dq_des[i] = A[i] * omega * std::cos(omega * t);
//             }
//

  // Maximum torque difference with a sampling rate of 1 kHz. The maximum torque rate is
  // 1000 * (1 / sampling_time).
  std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);  // Rate limit

  for (size_t i = 0; i < 7; ++i) {  // Loop over all joints
    joint_handles_[i].setCommand(tau_d_saturated[i]);  // Send torque command to joint
  }  // End torque command loop

  // torque_publisher_ is a ROS publisher used to publish the commanded joint torques (and sometimes
  // related diagnostics)
  //   at a throttled rate. It doesn’t control the robot directly; it just broadcasts torque values
  //   so other nodes or tools can monitor/plot/log
  //     them.
  //
  //       So conceptually:
  //
  //         - Controller output -> applied via joint_handles_[i].setCommand(tau)
  //           - Torque publisher -> publishes those torques for visibility/debugging, typically as
  //           a ROS topic (e.g., sensor_msgs/JointState or a custom
  //               message).
  //
  //
  if (rate_trigger_() && torques_publisher_.trylock()) {  // Publish torques at throttled rate
    std::array<double, 7> tau_j = robot_state.tau_J;  // Measured joint torques
    std::array<double, 7> tau_error;  // Error between commanded and measured torques
    double error_rms(0.0);  // RMS error accumulator
    for (size_t i = 0; i < 7; ++i) {  // Loop over all joints
      tau_error[i] = last_tau_d_[i] - tau_j[i];  // Per-joint torque error
      error_rms += std::sqrt(std::pow(tau_error[i], 2.0)) / 7.0;  // Accumulate RMS error
    }  // End error computation loop
    torques_publisher_.msg_.root_mean_square_error = error_rms;  // Store RMS error in message
    for (size_t i = 0; i < 7; ++i) {  // Loop over all joints
      torques_publisher_.msg_.tau_commanded[i] = last_tau_d_[i];  // Publish commanded torque
      torques_publisher_.msg_.tau_error[i] = tau_error[i];  // Publish torque error
      torques_publisher_.msg_.tau_measured[i] = tau_j[i];  // Publish measured torque
    }  // End message fill loop
    torques_publisher_.unlockAndPublish();  // Publish the message
  }  // End throttled publish block

  for (size_t i = 0; i < 7; ++i) {  // Loop over all joints
    last_tau_d_[i] = tau_d_saturated[i] + gravity[i];  // Cache torque with gravity for next cycle
  }  // End last torque update loop
}  // End update()

std::array<double, 7> JointImpedanceExampleController::saturateTorqueRate(
    const std::array<double, 7>& tau_d_calculated,
    const std::array<double, 7>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  std::array<double, 7> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointImpedanceExampleController,
                       controller_interface::ControllerBase)
