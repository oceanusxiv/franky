#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/robot_pose.hpp"
#include "franky/twist.hpp"
#include "franky/twist_acceleration.hpp"

namespace franky {

/**
 * @brief Cartesian impedance reference expressed in the base frame.
 */
struct CartesianReference {
  /** Desired end-effector pose. */
  Affine target{Affine::Identity()};

  /**
   * Desired end-effector twist in the base frame.
   *
   * When present, the damping term acts on twist error rather than resisting
   * all motion toward zero.
   */
  std::optional<Twist> target_twist{};

  /**
   * Desired end-effector acceleration in the base frame.
   *
   * When present, the controller adds a model-based inertial feedforward
   * wrench Lambda(q) * target_acceleration before mapping through J^T.
   */
  std::optional<TwistAcceleration> target_acceleration{};
};

/**
 * @brief Joint-posture objective projected into the Cartesian nullspace.
 */
struct PostureTask {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PostureTask() = default;

  PostureTask(
      const Vector7d &target, double stiffness, std::optional<double> damping = std::nullopt, double max_torque = 0.0)
      : target(target), stiffness(stiffness), damping(damping), max_torque(max_torque) {}

  /** Preferred joint posture [rad]. */
  Vector7d target{Vector7d::Zero()};

  /** Posture stiffness in [Nm/rad]. */
  double stiffness{0.0};

  /**
   * Posture damping in [Nms/rad].
   *
   * If unset, the controller uses critical damping, 2*sqrt(stiffness).
   */
  std::optional<double> damping{std::nullopt};

  /** Per-joint absolute torque clamp for this task [Nm]. Set <= 0 to disable. */
  double max_torque{0.0};
};

/**
 * @brief Manipulability maximization objective projected into the Cartesian nullspace.
 */
struct ManipulabilityTask {
  ManipulabilityTask() = default;

  ManipulabilityTask(double gain, double damping = 0.0, double max_torque = 0.0)
      : gain(gain), damping(damping), max_torque(max_torque) {}

  /** Gain applied to the manipulability gradient. */
  double gain{0.0};

  /** Joint damping applied to this task before projection [Nms/rad]. */
  double damping{0.0};

  /** Per-joint absolute torque clamp for this task [Nm]. Set <= 0 to disable. */
  double max_torque{0.0};
};

using NullspaceTask = std::variant<PostureTask, ManipulabilityTask>;

/**
 * @brief Runtime-adjustable gains for a nullspace task.
 *
 * The controller consumes the posture_* fields for PostureTask entries and the
 * manipulability_* fields for ManipulabilityTask entries. Keeping both sets in
 * one plain type avoids a second variant hierarchy in the realtime gains buffer.
 */
struct NullspaceGains {
  double posture_stiffness{0.0};
  std::optional<double> posture_damping{std::nullopt};
  double posture_max_torque{0.0};

  double manipulability_gain{0.0};
  double manipulability_damping{0.0};
  double manipulability_max_torque{0.0};
};

/**
 * @brief Double-buffered handle for updating nullspace task gains online.
 *
 * Constructed from a task vector to seed the initial gains. set() writes to
 * the inactive buffer and flips the active index; each controller consumes only
 * the named sections corresponding to its configured nullspace tasks.
 *
 * Thread safety: at most one thread may call set() or clear() at a time.
 * Concurrent reads from the RT callback via activeGains() and hasGains() are safe.
 */
class NullspaceGainsHandle {
 public:
  explicit NullspaceGainsHandle(const std::vector<NullspaceTask> &tasks);

  void set(const NullspaceGains &gains);
  void clear();
  [[nodiscard]] bool hasGains() const;
  [[nodiscard]] const NullspaceGains &activeGains() const;

 private:
  std::array<NullspaceGains, 2> buffers_{};
  std::atomic<uint8_t> active_index_{0};
  std::atomic<bool> valid_{false};
};

/**
 * @brief Cartesian impedance dynamics formulation.
 */
enum class CartesianImpedanceDynamicsMode {
  /** Direct Cartesian wrench spring-damper: tau = J^T F. */
  kWrench,

  /** Operational-space inertia decoupling: tau = J^T Lambda xdd_cmd. */
  kOperationalSpace,
};

/**
 * @brief Base class for client-side cartesian impedance motions.
 *
 * This motion is implements a cartesian impedance controller on the client
 * side and does not use Franka's internal impedance controller. Instead, it
 * uses Franka's internal torque controller and calculates the torques itself.
 */
class CartesianImpedanceBase : public Motion<franka::Torques> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Parameters for the impedance motion.
   */
  struct Params {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * Cartesian stiffness matrix [N/m, Nm/rad], expressed in the base frame with axis order
     * [x, y, z, rx, ry, rz] matching Model::zeroJacobian. Must be symmetric and positive
     * semi-definite. Defaults to the isotropic block-diagonal matrix corresponding to the
     * previous translational_stiffness=2000/rotational_stiffness=200 defaults; see
     * CartesianImpedanceGains::isotropic and CartesianImpedanceGains::diagonal for convenience
     * constructors covering the common non-coupled cases.
     */
    Matrix6d stiffness{defaultCartesianImpedanceStiffness()};

    /**
     * Cartesian damping matrix. nullopt → critical damping, see defaultCartesianImpedanceDamping.
     */
    std::optional<Matrix6d> damping{std::nullopt};

    /**
     * Maximum absolute Cartesian position error [m] used by the task-space controller.
     *
     * The translational error is clamped elementwise before the impedance wrench
     * is computed. This bounds the commanded Cartesian force when the reference
     * jumps or contact prevents the end effector from reaching the target.
     */
    Eigen::Vector3d translational_error_clip{Eigen::Vector3d::Constant(0.10)};

    /**
     * Maximum absolute Cartesian orientation error [rad] used by the task-space controller.
     *
     * The rotational error is clamped elementwise in the base frame before the
     * impedance wrench is computed. This bounds the commanded Cartesian torque.
     */
    Eigen::Vector3d rotational_error_clip{Eigen::Vector3d::Constant(0.25)};

    /** Per-axis force/torque constraints [N, Nm]. nullopt on an axis means unconstrained. */
    std::array<std::optional<double>, 6> force_constraints{};

    /** Cartesian task dynamics formulation. */
    CartesianImpedanceDynamicsMode dynamics_mode{CartesianImpedanceDynamicsMode::kWrench};

    /**
     * Nullspace objectives.
     *
     * Each task contributes a joint-space torque that is summed and projected
     * into the Jacobian nullspace.
     */
    std::vector<NullspaceTask> nullspace_tasks{};

    /** Shared torque safety limits and soft joint-limit repulsion settings. */
    TorqueSafetyParams safety{};

    /** Per-joint friction feedforward. Defaults to zero (disabled). */
    FrictionCompensationParams friction{};
  };

  /**
   * @brief Runtime update handles for long-lived Cartesian impedance motions.
   */
  struct RuntimeOptions {
    /** Optional handle for runtime Cartesian gain updates. */
    std::shared_ptr<CartesianImpedanceGainsHandle> gains_handle{};

    /** Optional handle for runtime nullspace task gain updates. */
    std::shared_ptr<NullspaceGainsHandle> nullspace_gains_handle{};

    /** Smoothing time constant for runtime gain transitions [s]. */
    double gains_time_constant{0.1};
  };

  /**
   * @param target The target pose.
   * @param params Parameters for the motion.
   * @param runtime Runtime update handles and smoothing configuration.
   */
  explicit CartesianImpedanceBase(Affine target, const Params &params);
  CartesianImpedanceBase(Affine target, const Params &params, RuntimeOptions runtime);

  [[nodiscard]] inline Affine target() const { return absolute_target_; }

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;

  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

  [[nodiscard]] inline Affine intermediate_target() const { return intermediate_target_; }

  [[nodiscard]] inline const Params &base_params() const { return params_; }

  inline void setAbsoluteTarget(const Affine &target) {
    target_ = target;
    absolute_target_ = target;
  }

  virtual std::tuple<CartesianReference, bool> update(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time,
      franka::Duration abs_time) = 0;

 private:
  void rebuildStiffnessDamping();

  Affine absolute_target_;
  Affine target_;
  Params params_;

  std::shared_ptr<CartesianImpedanceGainsHandle> gains_handle_;
  std::shared_ptr<NullspaceGainsHandle> nullspace_gains_handle_;
  double gains_time_constant_;
  Matrix6d current_stiffness_;
  std::optional<Matrix6d> current_damping_;
  NullspaceGains current_nullspace_gains_;

  Matrix6d stiffness, damping;
  Affine intermediate_target_;

  std::unique_ptr<franka::Model> model_;
};

using ImpedanceMotion = CartesianImpedanceBase;

}  // namespace franky
