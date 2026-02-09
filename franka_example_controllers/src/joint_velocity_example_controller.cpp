// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/joint_velocity_example_controller.h>

#include <cmath>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_example_controllers {

bool JointVelocityExampleController::init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {
  velocity_joint_interface_ = robot_hardware->get<hardware_interface::VelocityJointInterface>();
  if (velocity_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointVelocityExampleController: Error getting velocity joint interface from hardware!");
    return false;
  }

  std::string arm_id="panda";
  // if (!node_handle.getParam("arm_id", arm_id)) {
  //   ROS_ERROR("JointVelocityExampleController: Could not get parameter arm_id");
  //   return false;
  // }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointVelocityExampleController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointVelocityExampleController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }
  velocity_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      velocity_joint_handles_[i] = velocity_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointVelocityExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("JointVelocityExampleController: Could not get state interface from hardware");
    return false;
  }

  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(state_interface->getHandle(arm_id + "_robot"));

    for (size_t i = 0; i < q_start.size(); i++) {
      if (std::abs(state_handle_->getRobotState().q_d[i] - q_start[i]) > 0.1) {
        ROS_ERROR_STREAM(
            "JointVelocityExampleController: Robot is not in the expected starting position for "
            "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
            "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
        return false;
      }
    }
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "JointVelocityExampleController: Exception getting state handle: " << e.what());
    return false;
  }

  input_thread_ = std::thread([this]() {
      // It creates a new std::thread and immediately starts it. The thread runs a lambda that
      // captures this, so it can access the current object’s members.
      // This thread runs a loop (see the code just below) that blocks on getchar()
      // So the line is effectively “spawn" a background input thread attached to this controller instance.”
    while (ros::ok()) {
      // ros::ok() is a ROS utility that returns true while the node should keep running. It flips to
      // false when ROS is shutting down (e.g., Ctrl+C, node shutdown, master down), 
      // letting loops exit cleanly. In your code, it keeps the input thread alive until
      // ROS is shutting down.
      char c = getchar();   // or read GPIO / serial
      if (c == 'r') {       // press r to release
        release_requested_.store(true, std::memory_order_relaxed); // binary member variable
        // release_requested_ is std::atomic<bool> instead of a plain bool
        //
        //   - release_requested_ = true; calls store(true, std::memory_order_seq_cst) under the
        //   hood. That’s the strongest ordering (sequentially consistent).
        //   - release_requested_.store(true, std::memory_order_relaxed); stores the same value but with
        //   relaxed ordering (no synchronization guarantees beyond atomicity).
        //
        //   synchronization guarantee: when one thread writes the flag, what other memory writes
        //   are guaranteed to become visible to other threads when they see the flag change. 
        //
        //   So they both set the flag, but store(..., relaxed) is weaker and potentially faster. 
        //   For a simple “signal” flag with no data dependency, relaxed is usually fine. 
        //   If release_requested_ is a plain bool, then release_requested_ = true; is just a normal
        //   write and is not thread-safe.
        //
        // store is atomic but does not enforce any ordering or synchronization with other memory
        // operations.
        //
        //   Practical effects:
        //
        //    - The write to release_requested_ won’t be torn or partially observed.
        //    - But it does not guarantee when other threads see it relative to other
        //       writes/reads. It only guarantees that the atomic itself is updated.
        //
        //             Why it’s okay here:
        //
        //              - The flag is just a simple “signal” with no data dependency. The
        //               controller thread only needs to eventually see it as true; there’s no
        //                   requirement to synchronize other data with it.
        //              - If the program needed to ensure other data were visible when the
        //                     flag changes, it would use memory_order_release for the store and
        //                         memory_order_acquire on the load.
        //
        triggerRunBarrier();
        ROS_INFO("Release key pressed!");
      }
    }
  }); // lambda expression

  release_srv_ = node_handle.advertiseService("gripper_release", &JointVelocityExampleController::releaseServiceCallback, this);

  homing_client_ = std::make_unique<HomingClient>("/franka_gripper/homing", true);
  move_client_   = std::make_unique<MoveClient>  ("/franka_gripper/move",   true);
  grasp_client_  = std::make_unique<GraspClient> ("/franka_gripper/grasp",  true);
  stop_client_   = std::make_unique<StopClient>  ("/franka_gripper/stop",   true);

  homing_client_->waitForServer();
  move_client_->waitForServer();
  grasp_client_->waitForServer();
  stop_client_->waitForServer();

  franka_gripper::HomingGoal goal;
  homing_client_->sendGoal(goal);
  homing_client_->waitForResult();

  robot_reached_target_ = false;
  gripper_state_ = GripperState::OPEN;
  // This initializes the controller’s gripper state machine to the OPEN state. It means that when
  // the control loop later checks gripper_state_, it will treat the gripper as initially open 
  // and (if the arm has reached its target) it will send a grasp command next.
  //
  gripper_cmd_sent_ = false;
  release_requested_ = false;
  released_ = false;

  return true;
}

void JointVelocityExampleController::starting(const ros::Time& /* time */) {
  elapsed_time_ = ros::Duration(0.0);
}

void JointVelocityExampleController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {
  elapsed_time_ += period;

  // const std::array<double, 7> q_target{{2.207697,-1.271094,-1.800125,-1.124314,-0.074748,3.299907,-0.336494}}; //h=0.6
  // const std::array<double, 7> q_target{{2.252213,-1.153417,-1.879327,-1.054602,0.076281,3.239308,-0.347780}}; //h=0.65
  // const std::array<double, 7> q_target{{2.143326,-0.913033,-1.930531,-1.080025,-0.288167,3.091908,0.238997}}; //h=0.8
  const std::array<double, 7> q_target{{0.223019,0.337663,-0.289728,-1.041732,0.251676,2.935638,0.633736}}; //h=1.0

  ros::Duration time_max(8.0);

  const auto& robot_state = state_handle_->getRobotState();

  double max_e = 0.0;
  double max_dq = 0.0;

  for (size_t i = 0; i < 7; ++i) {
    double q = robot_state.q[i];                 // Current joint position for joint i (rad)
    double dq = robot_state.dq[i];               // Current joint velocity for joint i (rad/s)
    double e = q_target[i] - q;                  // Position error to target for joint i

    max_e = std::max(max_e, std::abs(e));        // Track maximum absolute position error over all joints
    max_dq = std::max(max_dq, std::abs(dq));     // Track maximum absolute velocity over all joints

    double omega_cmd = omega_max * std::tanh(kp_ * e); // Velocity command with smooth saturation
    // omega_max defined in header file, which is 0.2
    velocity_joint_handles_[i].setCommand(omega_cmd);  // Send commanded velocity to joint i
  }
  // ROS_INFO("max_e = %.6f, max_dq = %.6f", max_e,max_dq);

  // e_tol defined in header file, const double e_tol_ = 4e-2;    // rad
  // check if all joints reached target 
  if (max_e < e_tol_ && max_dq < dq_tol_) {      // Check if position and velocity are within tolerances
    stable_time_ += period;                      // Accumulate time spent inside tolerance
    if (stable_time_.toSec() > stable_duration_) { // If stable long enough, declare target reached
      robot_reached_target_ = true;              // Latch target reached flag
    }
  } else {
    stable_time_ = ros::Duration(0.0);           // Reset stable timer when out of tolerance
  }

  if (robot_reached_target_)                     // Only handle gripper once arm is at target
  {
    switch (gripper_state_)                      // Gripper state machine
    {
      case GripperState::OPEN:
        if (!gripper_cmd_sent_)                  // Send grasp command only once
        {
          franka_gripper::GraspGoal goal;        // Build grasp goal
          goal.width = 0.01;                     // Target grasp width
          // goal.width is the target finger opening width for the Franka gripper command. It’s the
          // distance between the two fingers, in meters.
          //
          //     - For GraspGoal, goal.width = 0.01 means “close until the gap is 1 cm.”
          //     - For MoveGoal, goal.width = 0.08 means “open to an 8 cm gap.”
          //
          // So it’s the desired gripper aperture.
          //
          goal.speed = 0.01;                     // Gripper closing speed
          goal.force = 4.0;                      // Grasping force
          grasp_client_->sendGoal(goal);         // Send grasp command

          gripper_cmd_sent_ = true;              // Remember we sent the grasp command
          gripper_state_ = GripperState::GRASP;  // Transition to GRASP state
        }
        break;

      case GripperState::GRASP:
        if (release_requested_)                  // Wait for external release signal
        {
          ROS_INFO("Release Signal Received!");  // Log release request
          gripper_state_ = GripperState::RELEASE; // Transition to RELEASE state
        }
        break;

      case GripperState::RELEASE:
        if (release_requested_ && !released_) {  // Send release command only once
          // stop_client_->sendGoal(franka_gripper::StopGoal());

          franka_gripper::MoveGoal goal;         // Build release goal
          goal.width = 0.08;                     // Target open width
          goal.speed = 0.5;                      // Opening speed
          move_client_->sendGoal(goal);          // Send open command
          released_ = true;                      // Remember release was sent
        }
        break;

      default:
        break;                                   // No action for other states
    }
  }
}                                                // End update()

bool JointVelocityExampleController::releaseServiceCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res)
{
    release_requested_ = true;

    res.success = true;
    res.message = "Gripper release requested";

    ROS_INFO("gripper_release service called");
    return true;
}

void JointVelocityExampleController::triggerRunBarrier() {
  int fd = open("/tmp/run_barrier", O_WRONLY | O_NONBLOCK);
  if (fd < 0) {
    ROS_WARN("Could not open /tmp/run_barrier");
    return;
  }
  ssize_t n = write(fd, " ", 1);
  if (n != 1) {
    ROS_WARN("Failed to write to /tmp/run_barrier");
  }
  close(fd);
}

void JointVelocityExampleController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointVelocityExampleController,
                       controller_interface::ControllerBase)
