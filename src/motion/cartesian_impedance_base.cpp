#include "franky/motion/cartesian_impedance_base.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "franky/model.hpp"
#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/robot_pose.hpp"

namespace franky {

namespace {

void rejectDuplicateRuntimeNullspaceTasks(const std::vector<NullspaceTask> &tasks) {
  bool has_posture = false;
  bool has_manipulability = false;
  for (const auto &task : tasks) {
    std::visit(
        [&](const auto &t) {
          using T = std::decay_t<decltype(t)>;
          if constexpr (std::is_same_v<T, PostureTask>) {
            if (has_posture) {
              throw std::invalid_argument("Runtime nullspace gains support at most one PostureTask");
            }
            has_posture = true;
          } else {
            if (has_manipulability) {
              throw std::invalid_argument("Runtime nullspace gains support at most one ManipulabilityTask");
            }
            has_manipulability = true;
          }
        },
        task);
  }
}

NullspaceGains nullspaceGainsFromTasks(const std::vector<NullspaceTask> &tasks) {
  NullspaceGains gains{};
  for (const auto &task : tasks) {
    std::visit(
        [&](const auto &t) {
          using T = std::decay_t<decltype(t)>;
          if constexpr (std::is_same_v<T, PostureTask>) {
            gains.posture_stiffness = t.stiffness;
            gains.posture_damping = t.damping;
            gains.posture_max_torque = t.max_torque;
          } else {
            gains.manipulability_gain = t.gain;
            gains.manipulability_damping = t.damping;
            gains.manipulability_max_torque = t.max_torque;
          }
        },
        task);
  }
  return gains;
}

template <typename T>
T lerp(const T &current, const T &target, double alpha) {
  return current + alpha * (target - current);
}

// default_current is only invoked when actually needed (target set, current unset) since it may be
// expensive (e.g. a matrix eigendecomposition) and this runs on the RT control path.
template <typename T, typename DefaultFn>
std::optional<T> lerpOptionalDamping(
    const std::optional<T> &current, const std::optional<T> &target, DefaultFn &&default_current, double alpha) {
  if (!target.has_value()) return std::nullopt;
  return lerp(current.has_value() ? *current : default_current(), *target, alpha);
}
}  // namespace

NullspaceGainsHandle::NullspaceGainsHandle(const std::vector<NullspaceTask> &tasks) : buffers_{} {
  const auto initial_gains = nullspaceGainsFromTasks(tasks);
  buffers_[0] = initial_gains;
  buffers_[1] = initial_gains;
}

void NullspaceGainsHandle::set(const NullspaceGains &gains) {
  const uint8_t next_index = 1 - active_index_.load(std::memory_order_relaxed);
  buffers_[next_index] = gains;
  active_index_.store(next_index, std::memory_order_release);
  valid_.store(true, std::memory_order_release);
}

void NullspaceGainsHandle::clear() { valid_.store(false, std::memory_order_release); }

bool NullspaceGainsHandle::hasGains() const { return valid_.load(std::memory_order_acquire); }

const NullspaceGains &NullspaceGainsHandle::activeGains() const {
  return buffers_[active_index_.load(std::memory_order_acquire)];
}

namespace {

// Returns {Lambda, M_inv_JT} where Lambda = (J M^{-1} J^T)^{-1} is the 6x6 task-space inertia
// matrix and M_inv_JT = M^{-1} J^T is kept for reuse in the dynamically consistent nullspace
// projector N = I - M_inv_JT * Lambda * J. SVD handles kinematic singularities gracefully.
std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 7, 6>> computeTaskSpaceInertia(
    const Jacobian &jacobian, const Eigen::Matrix<double, 7, 7> &M) {
  const Eigen::Matrix<double, 7, 6> M_inv_JT = M.ldlt().solve(jacobian.transpose());
  const Eigen::Matrix<double, 6, 6> J_Minv_JT = jacobian * M_inv_JT;
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(J_Minv_JT, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 6, 6> sigma_inv = Eigen::Matrix<double, 6, 6>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < 6; ++i) {
    if (svd.singularValues()[i] > tolerance) sigma_inv(i, i) = 1.0 / svd.singularValues()[i];
  }
  return {svd.matrixV() * sigma_inv * svd.matrixU().transpose(), M_inv_JT};
}

Eigen::Matrix<double, 6, 7> pseudoInverse(const Eigen::Matrix<double, 7, 6> &matrix) {
  Eigen::JacobiSVD<Eigen::Matrix<double, 7, 6>> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto &singular_values = svd.singularValues();
  Eigen::Matrix<double, 6, 7> sigma_pinv = Eigen::Matrix<double, 6, 7>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < singular_values.size(); ++i) {
    if (singular_values[i] > tolerance) sigma_pinv(i, i) = 1.0 / singular_values[i];
  }
  return svd.matrixV() * sigma_pinv * svd.matrixU().transpose();
}

Vector7d clampTorque(const Vector7d &tau, double max_torque) {
  if (max_torque <= 0.0) return tau;
  return tau.cwiseMax(-max_torque).cwiseMin(max_torque);
}

double manipulability(const Jacobian &jacobian) {
  const double determinant = (jacobian * jacobian.transpose()).determinant();
  return std::sqrt(std::max(determinant, 0.0));
}

// Analytical manipulability gradient using 7 FK pose calls
// Derivation: for a serial revolute chain, column k of the zero Jacobian is
//   Jk = [zk × (pn - pk); zk], so dJk/dqi (i <= k) = [(zi×zk)×r + zk×(zi×r); zi×zk]
// where r = pn - pk. For k < i the term is zero. The gradient follows from dw/dqi = w · tr(J† · dJ/dqi).
Vector7d manipulabilityGradient(const Model &model, const RobotState &robot_state, const Jacobian &jacobian) {
  const double w = manipulability(jacobian);
  if (w < 1e-10) return Vector7d::Zero();

  // Kinematic pseudoinverse J† = V Σ† Uᵀ via truncated SVD
  Eigen::JacobiSVD<Jacobian> svd(jacobian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 7, 6> J_pinv = Eigen::Matrix<double, 7, 6>::Zero();
  constexpr double tol = 1e-6;
  for (int i = 0; i < 6; ++i) {
    if (svd.singularValues()[i] > tol)
      J_pinv += (1.0 / svd.singularValues()[i]) * svd.matrixV().col(i) * svd.matrixU().col(i).transpose();
  }

  // Joint axes: bottom 3 rows of Jacobian (zk for column k)
  const auto z = jacobian.bottomRows<3>();

  // Joint origins from FK
  static constexpr std::array<franka::Frame, 7> kJointFrames = {
      franka::Frame::kJoint1,
      franka::Frame::kJoint2,
      franka::Frame::kJoint3,
      franka::Frame::kJoint4,
      franka::Frame::kJoint5,
      franka::Frame::kJoint6,
      franka::Frame::kJoint7};
  Eigen::Matrix<double, 3, 7> p;
  for (int k = 0; k < 7; ++k) p.col(k) = model.pose(kJointFrames[k], robot_state).translation();

  const Eigen::Vector3d pn = robot_state.O_T_EE.translation();

  Vector7d gradient = Vector7d::Zero();
  for (int i = 0; i < 7; ++i) {
    const Eigen::Vector3d zi = z.col(i);
    Eigen::Matrix<double, 6, 7> dJ = Eigen::Matrix<double, 6, 7>::Zero();
    for (int k = i; k < 7; ++k) {
      const Eigen::Vector3d zk = z.col(k);
      const Eigen::Vector3d r = pn - p.col(k);
      const Eigen::Vector3d zi_x_zk = zi.cross(zk);
      dJ.block<3, 1>(0, k) = zi_x_zk.cross(r) + zk.cross(zi.cross(r));
      dJ.block<3, 1>(3, k) = zi_x_zk;
    }
    // tr(J† · dJ/dqi) as elementwise product — avoids materializing the 7×7 product
    gradient[i] = w * (J_pinv.transpose().array() * dJ.array()).sum();
  }
  return gradient;
}

PostureTask applyGains(PostureTask task, const NullspaceGains &g) {
  task.stiffness = g.posture_stiffness;
  task.damping = g.posture_damping;
  task.max_torque = g.posture_max_torque;
  return task;
}

ManipulabilityTask applyGains(ManipulabilityTask task, const NullspaceGains &g) {
  task.gain = g.manipulability_gain;
  task.damping = g.manipulability_damping;
  task.max_torque = g.manipulability_max_torque;
  return task;
}

Vector7d computeTaskTorque(const PostureTask &task, const RobotState &robot_state) {
  const double stiffness = task.stiffness;
  if (stiffness <= 0.0) return Vector7d::Zero();
  const double damping = task.damping.value_or(2.0 * std::sqrt(stiffness));
  return clampTorque(stiffness * (task.target - robot_state.q) - damping * robot_state.dq, task.max_torque);
}

Vector7d computeTaskTorque(
    const ManipulabilityTask &task, const Model &model, const RobotState &robot_state, const Jacobian &jacobian) {
  if (task.gain == 0.0) return Vector7d::Zero();
  Vector7d tau = task.gain * manipulabilityGradient(model, robot_state, jacobian) - task.damping * robot_state.dq;
  return clampTorque(tau, task.max_torque);
}

}  // namespace

CartesianImpedanceBase::CartesianImpedanceBase(Affine target, const CartesianImpedanceBase::Params &params)
    : CartesianImpedanceBase(std::move(target), params, RuntimeOptions{}) {}

CartesianImpedanceBase::CartesianImpedanceBase(
    Affine target, const CartesianImpedanceBase::Params &params, CartesianImpedanceBase::RuntimeOptions runtime)
    : target_(std::move(target)),
      params_(params),
      gains_handle_(std::move(runtime.gains_handle)),
      nullspace_gains_handle_(std::move(runtime.nullspace_gains_handle)),
      gains_time_constant_(runtime.gains_time_constant),
      current_stiffness_(params.stiffness),
      current_damping_(params.damping),
      Motion<franka::Torques>() {
  if (nullspace_gains_handle_) rejectDuplicateRuntimeNullspaceTasks(params_.nullspace_tasks);
  current_nullspace_gains_ = nullspaceGainsFromTasks(params.nullspace_tasks);
  rebuildStiffnessDamping();
}

void CartesianImpedanceBase::rebuildStiffnessDamping() {
  stiffness = current_stiffness_;
  damping = current_damping_.has_value() ? *current_damping_ : defaultCartesianImpedanceDamping(current_stiffness_);
}

void CartesianImpedanceBase::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  auto robot_pose = Affine(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  intermediate_target_ = robot_pose;
  absolute_target_ = target_;
}

franka::Torques CartesianImpedanceBase::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  auto [reference, finish] = update(robot_state, time_step, rel_time, abs_time);
  intermediate_target_ = reference.target;

  const double dt = time_step.toSec();
  const double alpha = 1.0 - std::exp(-dt / gains_time_constant_);

  if (gains_handle_ && gains_handle_->hasGains()) {
    const auto target_gains = gains_handle_->get();
    current_stiffness_ = lerp(current_stiffness_, target_gains.stiffness, alpha);
    current_damping_ = lerpOptionalDamping(
        current_damping_,
        target_gains.damping,
        [&] { return defaultCartesianImpedanceDamping(current_stiffness_); },
        alpha);
    rebuildStiffnessDamping();
  }

  if (nullspace_gains_handle_ && nullspace_gains_handle_->hasGains()) {
    const auto &target = nullspace_gains_handle_->activeGains();
    auto &cur = current_nullspace_gains_;
    cur.posture_stiffness = lerp(cur.posture_stiffness, target.posture_stiffness, alpha);
    cur.posture_max_torque = lerp(cur.posture_max_torque, target.posture_max_torque, alpha);
    cur.posture_damping = lerpOptionalDamping(
        cur.posture_damping, target.posture_damping, [&] { return 2.0 * std::sqrt(cur.posture_stiffness); }, alpha);
    cur.manipulability_gain = lerp(cur.manipulability_gain, target.manipulability_gain, alpha);
    cur.manipulability_damping = lerp(cur.manipulability_damping, target.manipulability_damping, alpha);
    cur.manipulability_max_torque = lerp(cur.manipulability_max_torque, target.manipulability_max_torque, alpha);
  }

  auto model = robot()->model();
  Vector7d coriolis = model->coriolis(robot_state);
  Jacobian jacobian = model->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Quaterniond orientation(transform.rotation());

  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << robot_state.O_T_EE.translation() - intermediate_target_.translation();
  error.head(3) = error.head(3).cwiseMax(-params_.translational_error_clip).cwiseMin(params_.translational_error_clip);

  Eigen::Quaterniond quat(intermediate_target_.rotation());
  if (quat.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }

  Eigen::Quaterniond error_quaternion(orientation.inverse() * quat);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.linear() * error.tail(3);
  error.tail(3) = error.tail(3).cwiseMax(-params_.rotational_error_clip).cwiseMin(params_.rotational_error_clip);

  const Vector6d desired_twist =
      reference.target_twist.has_value() ? reference.target_twist->vector_repr() : Vector6d::Zero();
  const Vector6d measured_twist = jacobian * robot_state.dq;

  const Vector6d cartesian_feedback = -stiffness * error - damping * (measured_twist - desired_twist);
  Vector6d wrench_cartesian = cartesian_feedback;
  Eigen::Matrix<double, 6, 6> lambda = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 7, 6> M_inv_JT = Eigen::Matrix<double, 7, 6>::Zero();
  bool has_task_space_inertia = false;
  if (params_.dynamics_mode == CartesianImpedanceDynamicsMode::kOperationalSpace) {
    const Eigen::Matrix<double, 7, 7> M = model->mass(robot_state);
    std::tie(lambda, M_inv_JT) = computeTaskSpaceInertia(jacobian, M);
    has_task_space_inertia = true;
    wrench_cartesian = lambda * (cartesian_feedback + (reference.target_acceleration.has_value()
                                                           ? reference.target_acceleration->vector_repr()
                                                           : Vector6d::Zero()));
  } else if (reference.target_acceleration.has_value()) {
    const Eigen::Matrix<double, 7, 7> M = model->mass(robot_state);
    std::tie(lambda, M_inv_JT) = computeTaskSpaceInertia(jacobian, M);
    has_task_space_inertia = true;
    wrench_cartesian += lambda * reference.target_acceleration->vector_repr();
  }
  for (int i = 0; i < 6; ++i) {
    if (params_.force_constraints[i].has_value()) wrench_cartesian[i] = *params_.force_constraints[i];
  }

  auto tau_task = jacobian.transpose() * wrench_cartesian;
  Vector7d tau_nullspace = Vector7d::Zero();
  if (!params_.nullspace_tasks.empty()) {
    Eigen::Matrix<double, 7, 7> nullspace_projector;
    if (params_.dynamics_mode == CartesianImpedanceDynamicsMode::kOperationalSpace) {
      if (!has_task_space_inertia) {
        const Eigen::Matrix<double, 7, 7> M = model->mass(robot_state);
        std::tie(lambda, M_inv_JT) = computeTaskSpaceInertia(jacobian, M);
        has_task_space_inertia = true;
      }
      nullspace_projector = Eigen::Matrix<double, 7, 7>::Identity() - M_inv_JT * lambda * jacobian;
    } else {
      const auto jacobian_transpose_pinv = pseudoInverse(jacobian.transpose());
      nullspace_projector = Eigen::Matrix<double, 7, 7>::Identity() - jacobian.transpose() * jacobian_transpose_pinv;
    }
    Vector7d tau_nullspace_unprojected = Vector7d::Zero();
    for (const auto &task : params_.nullspace_tasks) {
      tau_nullspace_unprojected += std::visit(
          [&](const auto &concrete_task) -> Vector7d {
            using Task = std::decay_t<decltype(concrete_task)>;
            const auto effective = applyGains(concrete_task, current_nullspace_gains_);
            if constexpr (std::is_same_v<Task, ManipulabilityTask>) {
              return computeTaskTorque(effective, *model, robot_state, jacobian);
            } else {
              return computeTaskTorque(effective, robot_state);
            }
          },
          task);
    }
    if (params_.dynamics_mode == CartesianImpedanceDynamicsMode::kOperationalSpace) {
      tau_nullspace = nullspace_projector.transpose() * tau_nullspace_unprojected;
    } else {
      tau_nullspace = nullspace_projector * tau_nullspace_unprojected;
    }
  }

  Vector7d tau_limit = Vector7d::Zero();
  if (params_.safety.lower_joint_limits.has_value() && params_.safety.upper_joint_limits.has_value()) {
    tau_limit = franky::computeJointLimitTorque(
        robot_state.q,
        robot_state.dq,
        *params_.safety.lower_joint_limits,
        *params_.safety.upper_joint_limits,
        params_.safety.joint_limit_activation_distance,
        params_.safety.joint_limit_stiffness,
        params_.safety.joint_limit_damping,
        params_.safety.joint_limit_max_torque);
  }

  Vector7d tau_d = tau_task + tau_nullspace + tau_limit + coriolis;
  tau_d += computeFrictionCompensation(robot_state.dq, params_.friction);
  tau_d = franky::saturateTorqueRate(tau_d, robot_state.tau_J_d, params_.safety.max_delta_tau);

  std::array<double, 7> tau_d_array{};
  Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

  auto output = franka::Torques(tau_d_array);
  if (finish) output = franka::MotionFinished(output);

  return output;
}

}  // namespace franky
