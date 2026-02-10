// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_impedance_example_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka_example_controllers/pseudo_inversion.h>

namespace franka_example_controllers {

bool CartesianImpedanceExampleController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  sub_equilibrium_pose_ = node_handle.subscribe(  // Subscribe to pose target updates
      "equilibrium_pose",  // Topic name relative to this node handle namespace
      20,  // Queue size: buffer up to 20 messages if callbacks lag
      &CartesianImpedanceExampleController::equilibriumPoseCallback,  // Callback on new pose
      this,  // Callback is a member function on this instance
      ros::TransportHints().reliable().tcpNoDelay());  // TCP transport: reliable, no Nagle delay
  // Callback mechanism (how data flows):
  // - A publisher (e.g., RViz interactive marker or another node) sends PoseStamped messages to the
  //   "equilibrium_pose" topic.
  // - ROS delivers each message to this subscriber and invokes
  //   CartesianImpedanceExampleController::equilibriumPoseCallback(...).
  // - The callback typically locks a mutex and updates position_d_target_ / orientation_d_target_.
  // - The real-time update loop later reads those targets (under the same mutex) to compute torques.
  //
  // Queue behavior (what happens on lag):
  // - The queue stores up to 20 messages if callbacks are slower than incoming messages.
  // - If messages arrive faster than they are processed, the queue eventually fills.
  // - Once full, ROS drops older messages (so the callback sees the most recent pose, not every pose).
  // - This keeps the controller responsive to the latest target but can skip intermediate updates.
  //
  // In this example setup, the publisher is the interactive marker node:
  // - franka_example_controllers/scripts/interactive_marker.py publishes PoseStamped on
  //   "equilibrium_pose".
  // - RViz displays the marker; dragging it updates marker_pose in that script.
  // - A timer in the script publishes the pose at ~200 Hz, so the controller tracks the marker.
  //
  // Real-robot note:
  // - If you are not running RViz/interactive_marker, there may be no publisher at all.
  // - A human moving the end-effector does NOT publish equilibrium_pose messages.
  // - In that case the controller keeps the last target pose, and hand motion creates error
  //   the controller pushes against.

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CartesianImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "CartesianImpedanceExampleController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CartesianImpedanceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>(

      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&CartesianImpedanceExampleController::complianceParamCallback, this, _1, _2));

  position_d_.setZero();  // Eigen: sets all elements of the vector to 0
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  return true;
} // init

void CartesianImpedanceExampleController::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.rotation());

  // set nullspace equilibrium configuration to initial q
  q_d_nullspace_ = q_initial;
}

void CartesianImpedanceExampleController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& /*period*/) {
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  // coriolis_array is a 7‑element (7-DOF) vector of joint‑space Coriolis torques
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // jacobian_array is a vectorized 6×7 Jacobian in column‑major order (from FrankaModelInterface).
  // That means:
  //
  //   - 6 rows (Cartesian: 3 linear + 3 angular)
  //   - 7 columns (one per joint)
  //   - Stored column by column
  //
  //  So the layout is:
  //
  //  J = [ j11 j12 ... j_{1,7}
  //        j21 j22 ... j_{2,7}
  //        j31 j32 ... j_{3,7}
  //        j41 j42 ... j_{4,7}
  //        j51 j52 ... j_{5,7}
  //        j_{6,1} j_{6,2} ... j_{6,7} ]
  //
  //  jacobian_array = [j11 j21 j31 j41 j51 j61 j12 j22 j32 j42 j52 j_{6,2}  ...  
  //  j_{1,7} j27 j37 j47 j57 j_{6,7}]
  //
  //


  // convert to Eigen
  // Map:lightweight wrapper that views existing memory as an Eigen matrix/vector without copying.
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());

  // tau_J_d (often written tau_j_d) is the desired joint torque vector in franka::RobotState. It’s
  // the last torque command that the driver/controller intended to apply.
  //
  //   So:
  //
  //    - tau_J = measured joint torque
  //    - tau_J_d = desired/commanded joint torque
  //

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data())); 
  //  end-effector config matrix
  
  Eigen::Vector3d position(transform.translation());

  Eigen::Quaterniond orientation(transform.rotation());

  // compute error to desired pose
  // position error
  Eigen::Matrix<double, 6, 1> error;

  error.head(3) << position - position_d_;
  // set the first 3 entries of the 6‑vector error to the position error (x, y, z). The last
  // 3 entries (tail(3)) are used for the orientation error

  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }

  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // Transform to base frame
  error.tail(3) << -transform.rotation() * error.tail(3);

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);  // 7 DoF

  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // Cartesian PD control with damping ratio = 1
  tau_task << jacobian.transpose() *
                  (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));

  // nullspace PD control with damping ratio = 1
  // Jacobian means high dimension to low dimension, which only has right inverse
  // J*(I-J^{+}J) = J-J=0
  //
  // J^T maps low to high, only has left inverse: left inverse bring high dime to low
  // (J^T)^{+}(I-J^T(J^T)^{+}) = (J^T)^{+}-(J^T)^{+} = 0
  // desire q_0 can be 
  // - in proportional form \dot{q_0} = -k(q-q_{des})  
  // - gradient w.r.t. loss function: \dot{q_0}=-\nabla_q L(q,q_{des})
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * (q_d_nullspace_ - q) -
                        (2.0 * sqrt(nullspace_stiffness_)) * dq); // desired veclocity zero: damping

  // desired null space joint configuration: q_d_nullspace_ = q_initial;
  //
  // Desired torque
  tau_d << tau_task + tau_nullspace + coriolis + gravity;
  // Line 233 already includes gravity compensation indirectly by using the Coriolis term as
  // returned by Franka’s model handle — but not the explicit gravity vector.
  //
  //   In this controller, they intentionally add only:
  //
  //    tau_task + tau_nullspace + coriolis
  //
  //  and omit gravity. 
  //  This is a design choice: 
  //  - either this code relies on the robot’s internal model/low‑level control to 
  //  handle gravity
  //  - or they keep gravity out to make the impedance “feel” more compliant 
  //  (depending on mode).   
  //
  //*************When gravity not added*************************************************************
  // It won’t just “fall,” because the impedance terms (task‑space stiffness + nullspace stiffness)
  // generate torques that resist motion and hold the pose. That effectively counters gravity, 
  // but only as much as the gains allow. What you’ll see if gravity isn’t explicitly added:
  // - The arm may sag or settle to a slightly different equilibrium under gravity.
  // - The amount depends on stiffness/damping and payload.
  // - With low gains, the droop can be noticeable.
  //
  // So: no gravity term doesn’t mean free‑fall; it means the controller must “fight gravity” through 
  // stiffness, which can lead to steady‑state error unless gains are high.
  //
  //
  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRate(tau_d, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));
  }

  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by filtering
  cartesian_stiffness_ =
      filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;

  // the following variables are defined in
  // header:55  std::mutex position_and_orientation_d_target_mutex_;
  // header:56  Eigen::Vector3d position_d_target_;
  // header:57  Eigen::Quaterniond orientation_d_target_;
  // 
  //  A mutex doesn’t “belong” to data automatically. It’s just a lock object. You assign meaning by
  //  how you use it.
  //
  //    In this controller, position_and_orientation_d_target_mutex_ is used to protect
  //    position_d_target_ and orientation_d_target_:
  //
  //      - It is locked in update() before reading those targets.
  //        - It should also be locked in the callback that writes them (equilibriumPoseCallback
  //        / dynamic reconfigure).
  //
  //          So the correspondence is by convention: “this mutex guards these variables,” enforced
  //          by always locking it when accessing them. If you want, I can point to the exact
  //          callback
  //            lines where it’s locked on writes.
  //
  std::lock_guard<std::mutex> position_d_target_mutex_lock(  // Lock target pose updates
  // note position_d_target_mutex_lock is type std::lock_guard! not std::mutex!
      position_and_orientation_d_target_mutex_);  // Protects position_d_target_ and orientation_d_target_
  // Those lines use a mutex lock to safely read position_d_target_ and orientation_d_target_ while
  // another thread (e.g., dynamic reconfigure or interactive marker callback) might be updating them.
  //
  //     Mechanism:
  //
  //       - std::lock_guard<std::mutex> locks the mutex immediately.
  //       - It holds the lock for the rest of the scope.
  //        - When the scope ends, the lock is released automatically.
  //
  //  So the update loop sees a consistent target pose without data races.
  //
  // std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  // There are two variables:
  // 1. position_and_orientation_d_target_mutex_
  //    This is the mutex (the lockable object) that protects shared data.
  // 2. position_d_target_mutex_lock
  //    This is a lock_guard object. Its constructor locks the mutex, and its destructor unlocks it.
  // So one is the lock, the other is the thing being locked. The lock_guard “wraps” the mutex to make 
  // locking/unlocking automatic and exception‑safe.
  //
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  // position_d_ is the current desired end-effector position used by the controller.
  // It starts from the robot's pose at controller start and is filtered toward position_d_target_.
  //
  // The Cartesian PD law uses (position - position_d_) as the position error.
  // position_d_target_ is the raw target position received from the subscriber (e.g., interactive marker).
  // It is not used directly; it is low-pass filtered into position_d_ for smooth motion.
  // position_d_target_ is not random:
  // - It is initialized in init() with setZero().
  // - It is set to the current end-effector pose in starting().
  // - It is updated by the equilibriumPoseCallback when new target messages arrive.

  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceExampleController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void CartesianImpedanceExampleController::complianceParamCallback(
    franka_example_controllers::compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << 2.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();
  nullspace_stiffness_target_ = config.nullspace_stiffness;
}

void CartesianImpedanceExampleController::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);

  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  // position_d_target_ is not random:
  // - It is initialized in init() with setZero().
  // - It is set to the current end-effector pose in starting().
  // - This callback then updates it whenever a new equilibrium_pose message arrives.

  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceExampleController,
                       controller_interface::ControllerBase)
