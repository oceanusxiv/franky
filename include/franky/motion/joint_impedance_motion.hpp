#pragma once

#include <memory>
#include <optional>

#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/types.hpp"

namespace franky {

struct JointReference;

/**
 * @brief Client-side joint impedance controller.
 *
 * This motion uses Franka's torque interface and computes joint torques from a
 * joint-space spring-damper law plus optional torque offset/model compensation.
 */
struct JointImpedanceParams {
  /** Joint stiffness gains in [Nm/rad]. */
  Vector7d stiffness{defaultJointImpedanceStiffness()};

  /** Joint damping gains in [Nms/rad]. */
  Vector7d damping{defaultJointImpedanceDamping()};

  /** Constant torque offset added to every command in [Nm]. */
  Vector7d constant_torque_offset{Vector7d::Zero()};

  /** Compensate Coriolis forces using the robot model. */
  bool compensate_coriolis{true};

  /** Shared torque safety limits and soft joint-limit repulsion settings. */
  TorqueSafetyParams safety{};

  /** Joint friction compensation settings. */
  FrictionCompensationParams friction{};

  /**
   * Optional Cartesian-shaped gain term (see HybridCartesianGains).
   *
   * nullopt (default) disables the hybrid path entirely: Set to
   * enable a hybrid controller that adds J^T * diag(cartesian_stiffness) * J on top of
   * the joint stiffness/damping above, projecting Cartesian compliance into joint
   * space via the current Jacobian. Whether the hybrid path is active is fixed for the
   * lifetime of the motion; it cannot be toggled at runtime, only its values (via
   * RuntimeOptions::cartesian_gains_handle).
   */
  std::optional<HybridCartesianGains> cartesian_gains{std::nullopt};
};

class JointImpedanceBase : public Motion<franka::Torques> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Runtime update handles for long-lived joint impedance motions.
   */
  struct RuntimeOptions {
    /** Optional handle for runtime joint stiffness/damping updates. */
    std::shared_ptr<JointImpedanceGainsHandle> gains_handle{};

    /** Optional handle for runtime hybrid Cartesian gain updates. */
    std::shared_ptr<HybridCartesianGainsHandle> cartesian_gains_handle{};

    /** Smoothing time constant for runtime gain transitions [s]. */
    double gains_time_constant{0.1};
  };

  [[nodiscard]] const Vector7d &target() const { return target_; }
  [[nodiscard]] const Vector7d &target_velocity() const { return target_velocity_; }
  [[nodiscard]] const JointImpedanceParams &params() const { return params_; }

 protected:
  // Note: RuntimeOptions intentionally has no default argument on the constructor
  // below — a nested aggregate's default member initializers can't be used as a
  // default argument of an enclosing class's own member declared before the class is
  // complete (a hard compiler error, not just a style preference). Callers that don't
  // need runtime handles use the RuntimeOptions-less overload instead, which mirrors
  // CartesianImpedanceBase's two-constructor split.
  explicit JointImpedanceBase(
      const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params);
  JointImpedanceBase(
      const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params,
      RuntimeOptions runtime);

  [[nodiscard]] franka::Torques computeCommand(
      const RobotState &robot_state, const JointReference &reference, double dt);

  JointImpedanceParams params_;
  Vector7d target_;
  Vector7d target_velocity_;

 private:
  /** Resolved, continuously-smoothed hybrid Cartesian gain state. */
  struct HybridState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector6d stiffness;
    Vector6d damping;
  };

  std::shared_ptr<JointImpedanceGainsHandle> gains_handle_;
  std::shared_ptr<HybridCartesianGainsHandle> cartesian_gains_handle_;
  double gains_time_constant_;
  Vector7d current_stiffness_;
  Vector7d current_damping_;
  // nullopt <=> hybrid path disabled; presence is the only flag needed.
  std::optional<HybridState> hybrid_;
};

class JointImpedanceMotion : public JointImpedanceBase {
 public:
  using Params = JointImpedanceParams;

  explicit JointImpedanceMotion(const Vector7d &target);
  explicit JointImpedanceMotion(const Vector7d &target, const Params &params);
  JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity);
  JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity, const Params &params);

 protected:
  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;
};

}  // namespace franky
