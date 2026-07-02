#include "franky/motion/joint_impedance_motion.hpp"

#include <array>
#include <cmath>

#include "franky/model.hpp"
#include "franky/motion/joint_impedance_tracking_motion.hpp"
#include "franky/motion/torque_control_utils.hpp"

namespace franky {

JointImpedanceBase::JointImpedanceBase(
    const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params)
    : JointImpedanceBase(target, target_velocity, params, RuntimeOptions{}) {}

JointImpedanceBase::JointImpedanceBase(
    const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params, RuntimeOptions runtime)
    : Motion<franka::Torques>(),
      params_(params),
      target_(target),
      target_velocity_(target_velocity),
      gains_handle_(std::move(runtime.gains_handle)),
      cartesian_gains_handle_(std::move(runtime.cartesian_gains_handle)),
      gains_time_constant_(runtime.gains_time_constant),
      current_stiffness_(params.stiffness),
      current_damping_(params.damping) {
  if (params.cartesian_gains.has_value()) {
    const auto &cartesian_gains = *params.cartesian_gains;
    const Vector6d damping = cartesian_gains.damping.value_or(2.0 * cartesian_gains.stiffness.cwiseSqrt());
    hybrid_ = HybridState{cartesian_gains.stiffness, damping};
  }
}

JointImpedanceMotion::JointImpedanceMotion(const Vector7d &target) : JointImpedanceMotion(target, Params{}) {}

JointImpedanceMotion::JointImpedanceMotion(const Vector7d &target, const Params &params)
    : JointImpedanceMotion(target, Vector7d::Zero(), params) {}

JointImpedanceMotion::JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity)
    : JointImpedanceMotion(target, target_velocity, Params{}) {}

JointImpedanceMotion::JointImpedanceMotion(
    const Vector7d &target, const Vector7d &target_velocity, const Params &params)
    : JointImpedanceBase(target, target_velocity, params) {}

franka::Torques JointImpedanceMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  JointReference reference;
  reference.q = target_;
  reference.dq = target_velocity_;
  const double dt = time_step.toSec();
  return computeCommand(robot_state, reference, dt);
}

franka::Torques JointImpedanceBase::computeCommand(
    const RobotState &robot_state, const JointReference &reference, double dt) {
  // If a gains handle is present, interpolate toward the target gains.
  if (gains_handle_ && gains_handle_->hasGains()) {
    const auto target_gains = gains_handle_->get();
    const double alpha = 1.0 - std::exp(-dt / gains_time_constant_);
    current_stiffness_ += alpha * (target_gains.stiffness - current_stiffness_);
    current_damping_ += alpha * (target_gains.damping - current_damping_);
  }

  Vector7d torque_feedforward = params_.constant_torque_offset + reference.tau_ff;
  auto model = robot()->model();

  Vector7d tau_d;
  if (hybrid_.has_value()) {
    // If a gains handle is present, interpolate toward the target Cartesian gains.
    if (cartesian_gains_handle_ && cartesian_gains_handle_->hasGains()) {
      const auto target_gains = cartesian_gains_handle_->get();
      const double alpha = 1.0 - std::exp(-dt / gains_time_constant_);
      hybrid_->stiffness += alpha * (target_gains.stiffness - hybrid_->stiffness);
      const Vector6d target_damping = target_gains.damping.value_or(2.0 * target_gains.stiffness.cwiseSqrt());
      hybrid_->damping += alpha * (target_damping - hybrid_->damping);
    }

    const Jacobian jacobian = model->zeroJacobian(franka::Frame::kEndEffector, robot_state);
    const Eigen::Matrix<double, 7, 7> stiffness_eff = current_stiffness_.asDiagonal().toDenseMatrix() +
                                                      jacobian.transpose() * hybrid_->stiffness.asDiagonal() * jacobian;
    const Eigen::Matrix<double, 7, 7> damping_eff =
        current_damping_.asDiagonal().toDenseMatrix() + jacobian.transpose() * hybrid_->damping.asDiagonal() * jacobian;
    tau_d = stiffness_eff * (reference.q - robot_state.q) + damping_eff * (reference.dq - robot_state.dq) +
            torque_feedforward;
  } else {
    tau_d = current_stiffness_.asDiagonal() * (reference.q - robot_state.q) +
            current_damping_.asDiagonal() * (reference.dq - robot_state.dq) + torque_feedforward;
  }

  tau_d += computeFrictionCompensation(robot_state.dq, params_.friction);

  if (params_.safety.lower_joint_limits.has_value() && params_.safety.upper_joint_limits.has_value()) {
    tau_d += franky::computeJointLimitTorque(
        robot_state.q,
        robot_state.dq,
        *params_.safety.lower_joint_limits,
        *params_.safety.upper_joint_limits,
        params_.safety.joint_limit_activation_distance,
        params_.safety.joint_limit_stiffness,
        params_.safety.joint_limit_damping,
        params_.safety.joint_limit_max_torque);
  }

  if (params_.compensate_coriolis) tau_d += model->coriolis(robot_state);
  tau_d = franky::saturateTorqueRate(tau_d, robot_state.tau_J_d, params_.safety.max_delta_tau);

  std::array<double, 7> tau_d_array{};
  Eigen::VectorXd::Map(tau_d_array.data(), 7) = tau_d;
  return franka::Torques(tau_d_array);
}

}  // namespace franky
