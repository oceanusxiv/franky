#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

namespace {

CartesianReference toCartesianReference(
    const Affine &target, const std::optional<Twist> &target_twist,
    const std::optional<TwistAcceleration> &target_acceleration) {
  CartesianReference reference;
  reference.target = target;
  reference.target_twist = target_twist;
  reference.target_acceleration = target_acceleration;
  return reference;
}

JointReference toJointReference(
    const Vector7d &position, const std::optional<Vector7d> &velocity,
    const std::optional<Vector7d> &torque_feedforward) {
  JointReference reference;
  reference.q = position;
  if (velocity.has_value()) reference.dq = velocity.value();
  if (torque_feedforward.has_value()) reference.tau_ff = torque_feedforward.value();
  return reference;
}

void validateNonNegativeFinite(const Vector7d &values, const char *name) {
  for (int i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i]) || values[i] < 0.0) {
      throw py::value_error(std::string(name) + " must contain only finite, non-negative values");
    }
  }
}

void validateNonNegativeFinite(const Vector6d &values, const char *name) {
  for (int i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i]) || values[i] < 0.0) {
      throw py::value_error(std::string(name) + " must contain only finite, non-negative values");
    }
  }
}

void validateFrictionCompensationParams(const FrictionCompensationParams &friction) {
  validateNonNegativeFinite(friction.coulomb, "friction.coulomb");
  validateNonNegativeFinite(friction.viscous, "friction.viscous");
  validateNonNegativeFinite(friction.max_torque, "friction.max_torque");
  if (friction.velocity_epsilon <= 0.0 || !std::isfinite(friction.velocity_epsilon)) {
    throw py::value_error("friction.velocity_epsilon must be finite and positive");
  }
}

void validateFinite(double value, const char *name) {
  if (!std::isfinite(value)) throw py::value_error(std::string(name) + " must be finite");
}

void validateNonNegativeFinite(double value, const char *name) {
  validateFinite(value, name);
  if (value < 0.0) throw py::value_error(std::string(name) + " must be non-negative");
}

void validateSymmetricPSD(const Matrix6d &matrix, const char *name) {
  if (!matrix.allFinite()) throw py::value_error(std::string(name) + " must contain only finite values");
  constexpr double tolerance = 1e-6;
  if ((matrix - matrix.transpose()).cwiseAbs().maxCoeff() > tolerance) {
    throw py::value_error(std::string(name) + " must be symmetric");
  }
  Eigen::SelfAdjointEigenSolver<Matrix6d> solver(matrix, Eigen::EigenvaluesOnly);
  if (solver.eigenvalues().minCoeff() < -tolerance) {
    throw py::value_error(std::string(name) + " must be positive semi-definite");
  }
}

// Accepted forms for the stiffness/damping kwargs: a full 6x6 matrix, or a 6-vector of per-axis
// (x, y, z, rx, ry, rz) gains with no cross-axis coupling (equivalent to matrix.asDiagonal()).
using CartesianGainInput = std::variant<Vector6d, Matrix6d>;

Matrix6d toCartesianGainMatrix(const CartesianGainInput &input) {
  if (std::holds_alternative<Matrix6d>(input)) return std::get<Matrix6d>(input);
  return std::get<Vector6d>(input).asDiagonal();
}

// Resolves the stiffness/damping kwargs (matrix or 6-vector) against the legacy isotropic scalar
// kwargs. At most one of {stiffness, {translational,rotational}_stiffness} may be given, same for
// damping. translational_stiffness/rotational_stiffness default to 2000.0/200.0 when neither form
// is given.
CartesianImpedanceGains resolveCartesianImpedanceGains(
    const std::optional<CartesianGainInput> &stiffness, const std::optional<CartesianGainInput> &damping,
    std::optional<double> translational_stiffness, std::optional<double> rotational_stiffness,
    std::optional<double> translational_damping, std::optional<double> rotational_damping) {
  if (stiffness.has_value() && (translational_stiffness.has_value() || rotational_stiffness.has_value())) {
    throw py::value_error("Provide either 'stiffness' or 'translational_stiffness'/'rotational_stiffness', not both");
  }
  if (damping.has_value() && (translational_damping.has_value() || rotational_damping.has_value())) {
    throw py::value_error("Provide either 'damping' or 'translational_damping'/'rotational_damping', not both");
  }
  if (stiffness.has_value() && !damping.has_value() &&
      (translational_damping.has_value() || rotational_damping.has_value())) {
    throw py::value_error(
        "'translational_damping'/'rotational_damping' derive their defaults from 'translational_stiffness'/"
        "'rotational_stiffness'; provide 'damping' as a full matrix (or omit it for critical damping) when "
        "'stiffness' is given as a matrix");
  }
  CartesianImpedanceGains gains;
  if (stiffness.has_value()) {
    const Matrix6d matrix = toCartesianGainMatrix(*stiffness);
    validateSymmetricPSD(matrix, "stiffness");
    gains.stiffness = matrix;
  } else {
    gains = CartesianImpedanceGains::isotropic(
        translational_stiffness.value_or(2000.0),
        rotational_stiffness.value_or(200.0),
        translational_damping,
        rotational_damping);
  }
  if (damping.has_value()) {
    const Matrix6d matrix = toCartesianGainMatrix(*damping);
    validateSymmetricPSD(matrix, "damping");
    gains.damping = matrix;
  }
  return gains;
}

std::vector<NullspaceTask> toNullspaceTasks(const py::object &object) {
  std::vector<NullspaceTask> tasks;
  if (object.is_none()) return tasks;
  for (const auto item : object) {
    if (py::isinstance<PostureTask>(item)) {
      tasks.emplace_back(item.cast<PostureTask>());
    } else if (py::isinstance<ManipulabilityTask>(item)) {
      tasks.emplace_back(item.cast<ManipulabilityTask>());
    } else {
      throw py::type_error("nullspace_tasks must contain only PostureTask or ManipulabilityTask instances");
    }
  }
  return tasks;
}

JointImpedanceParams makeJointImpedanceParams(
    const std::optional<Vector7d> &stiffness, const std::optional<Vector7d> &damping,
    const std::optional<Vector7d> &constant_torque_offset, const std::optional<Vector7d> &lower_joint_limits,
    const std::optional<Vector7d> &upper_joint_limits, bool compensate_coriolis, double max_delta_tau,
    double joint_limit_activation_distance, double joint_limit_stiffness, double joint_limit_damping,
    double joint_limit_max_torque, const std::optional<FrictionCompensationParams> &friction,
    const std::optional<Vector6d> &cartesian_stiffness, const std::optional<Vector6d> &cartesian_damping) {
  auto params = JointImpedanceParams{};
  if (stiffness.has_value()) {
    validateNonNegativeFinite(stiffness.value(), "stiffness");
    params.stiffness = stiffness.value();
    if (!damping.has_value()) params.damping = defaultJointImpedanceDamping(params.stiffness);
  }
  if (damping.has_value()) {
    validateNonNegativeFinite(damping.value(), "damping");
    params.damping = damping.value();
  }
  if (friction.has_value()) {
    validateFrictionCompensationParams(*friction);
    params.friction = *friction;
  }
  if (constant_torque_offset.has_value()) params.constant_torque_offset = constant_torque_offset.value();
  if (cartesian_stiffness.has_value()) {
    validateNonNegativeFinite(cartesian_stiffness.value(), "cartesian_stiffness");
    HybridCartesianGains cartesian_gains;
    cartesian_gains.stiffness = cartesian_stiffness.value();
    if (cartesian_damping.has_value()) {
      validateNonNegativeFinite(cartesian_damping.value(), "cartesian_damping");
      cartesian_gains.damping = cartesian_damping.value();
    }
    params.cartesian_gains = cartesian_gains;
  } else if (cartesian_damping.has_value()) {
    throw py::value_error("cartesian_damping requires cartesian_stiffness to be set");
  }
  params.safety.lower_joint_limits = lower_joint_limits;
  params.safety.upper_joint_limits = upper_joint_limits;
  params.compensate_coriolis = compensate_coriolis;
  params.safety.max_delta_tau = max_delta_tau;
  params.safety.joint_limit_activation_distance = joint_limit_activation_distance;
  params.safety.joint_limit_stiffness = joint_limit_stiffness;
  params.safety.joint_limit_damping = joint_limit_damping;
  params.safety.joint_limit_max_torque = joint_limit_max_torque;
  return params;
}

CartesianImpedanceBase::Params makeCartesianImpedanceParams(
    const std::optional<CartesianGainInput> &stiffness, const std::optional<CartesianGainInput> &damping,
    std::optional<double> translational_stiffness, std::optional<double> rotational_stiffness,
    std::optional<double> translational_damping, std::optional<double> rotational_damping,
    const std::optional<std::array<std::optional<double>, 6>> &force_constraints, double max_delta_tau,
    const std::optional<Vector7d> &lower_joint_limits, const std::optional<Vector7d> &upper_joint_limits,
    double joint_limit_activation_distance, double joint_limit_stiffness, double joint_limit_damping,
    double joint_limit_max_torque, const Eigen::Vector3d &translational_error_clip,
    const Eigen::Vector3d &rotational_error_clip, const std::vector<NullspaceTask> &nullspace_tasks,
    const std::optional<FrictionCompensationParams> &friction) {
  auto params = CartesianImpedanceBase::Params{};
  const auto gains = resolveCartesianImpedanceGains(
      stiffness, damping, translational_stiffness, rotational_stiffness, translational_damping, rotational_damping);
  params.stiffness = gains.stiffness;
  params.damping = gains.damping;
  params.translational_error_clip = translational_error_clip;
  params.rotational_error_clip = rotational_error_clip;
  params.safety.max_delta_tau = max_delta_tau;
  params.safety.joint_limit_activation_distance = joint_limit_activation_distance;
  params.safety.joint_limit_stiffness = joint_limit_stiffness;
  params.safety.joint_limit_damping = joint_limit_damping;
  params.safety.joint_limit_max_torque = joint_limit_max_torque;
  params.safety.lower_joint_limits = lower_joint_limits;
  params.safety.upper_joint_limits = upper_joint_limits;
  params.nullspace_tasks = nullspace_tasks;
  if (force_constraints.has_value()) params.force_constraints = force_constraints.value();
  if (friction.has_value()) {
    validateFrictionCompensationParams(*friction);
    params.friction = *friction;
  }
  return params;
}

}  // namespace

void bind_motion_torque(py::module &m) {
  auto cartesian_impedance_base =
      py::class_<CartesianImpedanceBase, Motion<franka::Torques>, std::shared_ptr<CartesianImpedanceBase>>(
          m, "CartesianImpedanceBase");
  m.attr("ImpedanceMotion") = cartesian_impedance_base;

  py::class_<PostureTask>(m, "PostureTask")
      .def(
          py::init<>([](const std::array<double, 7> &target,
                        double stiffness,
                        std::optional<double>
                            damping,
                        double max_torque) {
            validateNonNegativeFinite(stiffness, "stiffness");
            if (damping.has_value()) validateNonNegativeFinite(damping.value(), "damping");
            validateNonNegativeFinite(max_torque, "max_torque");
            return PostureTask{Eigen::Map<const Vector7d>(target.data()), stiffness, damping, max_torque};
          }),
          "target"_a,
          "stiffness"_a,
          "damping"_a = std::nullopt,
          "max_torque"_a = 0.0)
      .def_readwrite("target", &PostureTask::target)
      .def_readwrite("stiffness", &PostureTask::stiffness)
      .def_readwrite("damping", &PostureTask::damping)
      .def_readwrite("max_torque", &PostureTask::max_torque);

  py::class_<ManipulabilityTask>(m, "ManipulabilityTask")
      .def(
          py::init<>([](double gain, double damping, double max_torque) {
            validateFinite(gain, "gain");
            validateNonNegativeFinite(damping, "damping");
            validateNonNegativeFinite(max_torque, "max_torque");
            return ManipulabilityTask{gain, damping, max_torque};
          }),
          "gain"_a,
          "damping"_a = 0.0,
          "max_torque"_a = 0.0)
      .def_readwrite("gain", &ManipulabilityTask::gain)
      .def_readwrite("damping", &ManipulabilityTask::damping)
      .def_readwrite("max_torque", &ManipulabilityTask::max_torque);

  py::class_<NullspaceGains>(m, "NullspaceGains")
      .def(
          py::init<>([](double posture_stiffness,
                        std::optional<double>
                            posture_damping,
                        double posture_max_torque,
                        double manipulability_gain,
                        double manipulability_damping,
                        double manipulability_max_torque) {
            validateNonNegativeFinite(posture_stiffness, "posture_stiffness");
            if (posture_damping.has_value()) validateNonNegativeFinite(posture_damping.value(), "posture_damping");
            validateNonNegativeFinite(posture_max_torque, "posture_max_torque");
            validateFinite(manipulability_gain, "manipulability_gain");
            validateNonNegativeFinite(manipulability_damping, "manipulability_damping");
            validateNonNegativeFinite(manipulability_max_torque, "manipulability_max_torque");
            return NullspaceGains{
                posture_stiffness,
                posture_damping,
                posture_max_torque,
                manipulability_gain,
                manipulability_damping,
                manipulability_max_torque};
          }),
          "posture_stiffness"_a = 0.0,
          "posture_damping"_a = std::nullopt,
          "posture_max_torque"_a = 0.0,
          "manipulability_gain"_a = 0.0,
          "manipulability_damping"_a = 0.0,
          "manipulability_max_torque"_a = 0.0)
      .def_readwrite("posture_stiffness", &NullspaceGains::posture_stiffness)
      .def_readwrite("posture_damping", &NullspaceGains::posture_damping)
      .def_readwrite("posture_max_torque", &NullspaceGains::posture_max_torque)
      .def_readwrite("manipulability_gain", &NullspaceGains::manipulability_gain)
      .def_readwrite("manipulability_damping", &NullspaceGains::manipulability_damping)
      .def_readwrite("manipulability_max_torque", &NullspaceGains::manipulability_max_torque);

  py::class_<NullspaceGainsHandle, std::shared_ptr<NullspaceGainsHandle>>(m, "NullspaceGainsHandle")
      .def(
          py::init<>([](const py::object &nullspace_tasks) {
            return std::make_shared<NullspaceGainsHandle>(toNullspaceTasks(nullspace_tasks));
          }),
          R"doc(Construct a NullspaceGainsHandle for the given task list.

The task list seeds the initial gains. When attached to a motion, only fields corresponding
to that motion's configured nullspace tasks are consumed.)doc",
          "nullspace_tasks"_a)
      .def(
          "set",
          [](NullspaceGainsHandle &handle, const NullspaceGains &gains) { handle.set(gains); },
          R"doc(Update nullspace task gains for a running motion.

Only the fields corresponding to configured nullspace task kinds are consumed by the controller.)doc",
          "gains"_a)
      .def("clear", &NullspaceGainsHandle::clear)
      .def_property_readonly("has_gains", &NullspaceGainsHandle::hasGains);

  py::class_<CartesianImpedanceGains>(m, "CartesianImpedanceGains")
      .def(
          py::init<>([](std::optional<CartesianGainInput> stiffness,
                        std::optional<CartesianGainInput>
                            damping,
                        std::optional<double>
                            translational_stiffness,
                        std::optional<double>
                            rotational_stiffness,
                        std::optional<double>
                            translational_damping,
                        std::optional<double>
                            rotational_damping) {
            return resolveCartesianImpedanceGains(
                stiffness,
                damping,
                translational_stiffness,
                rotational_stiffness,
                translational_damping,
                rotational_damping);
          }),
          R"doc(Cartesian impedance gains in the base frame ([x, y, z, rx, ry, rz] order).

Provide either `stiffness` (a symmetric PSD 6x6 matrix, or a 6-vector for per-axis gains with no
cross-axis coupling) or the legacy `translational_stiffness`/`rotational_stiffness` scalars (which
build an isotropic block-diagonal matrix), not both. Same for `damping`, which defaults to critical
damping (2*sqrt(stiffness) generalized via the matrix square root) when omitted.

See also CartesianImpedanceGains.isotropic and CartesianImpedanceGains.diagonal for convenience constructors.)doc",
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "translational_stiffness"_a = std::nullopt,
          "rotational_stiffness"_a = std::nullopt,
          "translational_damping"_a = std::nullopt,
          "rotational_damping"_a = std::nullopt)
      .def_readwrite("stiffness", &CartesianImpedanceGains::stiffness)
      .def_readwrite("damping", &CartesianImpedanceGains::damping)
      .def_static(
          "isotropic",
          [](double translational_stiffness,
             double rotational_stiffness,
             std::optional<double>
                 translational_damping,
             std::optional<double>
                 rotational_damping) {
            auto gains = CartesianImpedanceGains::isotropic(
                translational_stiffness, rotational_stiffness, translational_damping, rotational_damping);
            validateSymmetricPSD(gains.stiffness, "stiffness");
            if (gains.damping.has_value()) validateSymmetricPSD(*gains.damping, "damping");
            return gains;
          },
          "Convenience constructor for independent translational/rotational stiffness with no axis coupling.",
          "translational_stiffness"_a,
          "rotational_stiffness"_a,
          "translational_damping"_a = std::nullopt,
          "rotational_damping"_a = std::nullopt)
      .def_static(
          "diagonal",
          [](const Vector6d &stiffness, std::optional<Vector6d> damping) {
            auto gains = CartesianImpedanceGains::diagonal(stiffness, damping);
            validateSymmetricPSD(gains.stiffness, "stiffness");
            if (gains.damping.has_value()) validateSymmetricPSD(*gains.damping, "damping");
            return gains;
          },
          "Convenience constructor for per-axis (x, y, z, rx, ry, rz) stiffness with no cross-axis coupling.",
          "stiffness"_a,
          "damping"_a = std::nullopt);

  py::class_<CartesianImpedanceGainsHandle, std::shared_ptr<CartesianImpedanceGainsHandle>>(
      m, "CartesianImpedanceGainsHandle")
      .def(py::init<>())
      .def(
          "set",
          [](CartesianImpedanceGainsHandle &handle, const CartesianImpedanceGains &gains) {
            validateSymmetricPSD(gains.stiffness, "stiffness");
            if (gains.damping.has_value()) validateSymmetricPSD(*gains.damping, "damping");
            handle.set(gains);
          },
          "gains"_a)
      .def("clear", &CartesianImpedanceGainsHandle::clear)
      .def("get", &CartesianImpedanceGainsHandle::get)
      .def_property_readonly("has_gains", &CartesianImpedanceGainsHandle::hasGains);

  py::class_<JointImpedanceGains>(m, "JointImpedanceGains")
      .def(
          py::init<>([](const std::optional<Vector7d> &stiffness, const std::optional<Vector7d> &damping) {
            JointImpedanceGains g;
            if (stiffness.has_value()) {
              validateNonNegativeFinite(stiffness.value(), "stiffness");
              g.stiffness = stiffness.value();
              if (!damping.has_value()) g.damping = defaultJointImpedanceDamping(g.stiffness);
            }
            if (damping.has_value()) {
              validateNonNegativeFinite(damping.value(), "damping");
              g.damping = damping.value();
            }
            return g;
          }),
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt)
      .def_readwrite("stiffness", &JointImpedanceGains::stiffness)
      .def_readwrite("damping", &JointImpedanceGains::damping);

  py::class_<JointImpedanceGainsHandle, std::shared_ptr<JointImpedanceGainsHandle>>(m, "JointImpedanceGainsHandle")
      .def(py::init<>())
      .def(
          "set",
          [](JointImpedanceGainsHandle &handle, const Vector7d &stiffness, const Vector7d &damping) {
            handle.set(JointImpedanceGains{stiffness, damping});
          },
          "stiffness"_a,
          "damping"_a)
      .def("clear", &JointImpedanceGainsHandle::clear)
      .def("get", &JointImpedanceGainsHandle::get)
      .def_property_readonly("has_gains", &JointImpedanceGainsHandle::hasGains);

  py::class_<HybridCartesianGains>(m, "HybridCartesianGains")
      .def(
          py::init<>([](const std::optional<Vector6d> &stiffness, const std::optional<Vector6d> &damping) {
            HybridCartesianGains g;
            if (stiffness.has_value()) {
              validateNonNegativeFinite(stiffness.value(), "stiffness");
              g.stiffness = stiffness.value();
            }
            if (damping.has_value()) {
              validateNonNegativeFinite(damping.value(), "damping");
              g.damping = damping.value();
            }
            return g;
          }),
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt)
      .def_readwrite("stiffness", &HybridCartesianGains::stiffness)
      .def_readwrite("damping", &HybridCartesianGains::damping);

  py::class_<HybridCartesianGainsHandle, std::shared_ptr<HybridCartesianGainsHandle>>(m, "HybridCartesianGainsHandle")
      .def(py::init<>())
      .def(
          "set",
          [](HybridCartesianGainsHandle &handle, const Vector6d &stiffness, const std::optional<Vector6d> &damping) {
            validateNonNegativeFinite(stiffness, "stiffness");
            HybridCartesianGains gains;
            gains.stiffness = stiffness;
            if (damping.has_value()) {
              validateNonNegativeFinite(damping.value(), "damping");
              gains.damping = damping.value();
            }
            handle.set(gains);
          },
          "stiffness"_a,
          "damping"_a = std::nullopt)
      .def("clear", &HybridCartesianGainsHandle::clear)
      .def("get", &HybridCartesianGainsHandle::get)
      .def_property_readonly("has_gains", &HybridCartesianGainsHandle::hasGains);

  py::class_<JointReferenceHandle, std::shared_ptr<JointReferenceHandle>>(m, "JointReferenceHandle")
      .def(py::init<>())
      .def(
          "set",
          [](JointReferenceHandle &handle,
             const Vector7d &position,
             std::optional<Vector7d>
                 velocity,
             std::optional<Vector7d>
                 torque_feedforward) { handle.set(toJointReference(position, velocity, torque_feedforward)); },
          R"doc(Update the current joint reference for a running JointImpedanceTrackingMotion.

The provided torque_feedforward is added on top of any constant_torque_offset configured on the motion.)doc",
          "position"_a,
          "velocity"_a = std::nullopt,
          "torque_feedforward"_a = std::nullopt)
      .def("clear", &JointReferenceHandle::clear)
      .def_property_readonly("has_reference", &JointReferenceHandle::hasReference);

  py::class_<CartesianReferenceHandle, std::shared_ptr<CartesianReferenceHandle>>(m, "CartesianReferenceHandle")
      .def(py::init<>())
      .def(
          "set",
          [](CartesianReferenceHandle &handle,
             const Affine &target,
             std::optional<Twist>
                 target_twist,
             std::optional<TwistAcceleration>
                 target_acceleration) { handle.set(toCartesianReference(target, target_twist, target_acceleration)); },
          R"doc(Update the current Cartesian reference for a running CartesianImpedanceTrackingMotion.

If target_twist is provided, it is interpreted as the desired end-effector twist in the base frame and the damping term acts on twist error rather than motion relative to zero.

If target_acceleration is provided, it is interpreted as the desired end-effector acceleration in the base frame and contributes a model-based inertial feedforward wrench.)doc",
          "target"_a,
          "target_twist"_a = std::nullopt,
          "target_acceleration"_a = std::nullopt)
      .def("clear", &CartesianReferenceHandle::clear)
      .def_property_readonly("has_reference", &CartesianReferenceHandle::hasReference);

  // Params classes — bind before the motion classes that use them.

  py::class_<TorqueSafetyParams>(m, "TorqueSafetyParams")
      .def(py::init<>())
      .def_readwrite("max_delta_tau", &TorqueSafetyParams::max_delta_tau)
      .def_readwrite("lower_joint_limits", &TorqueSafetyParams::lower_joint_limits)
      .def_readwrite("upper_joint_limits", &TorqueSafetyParams::upper_joint_limits)
      .def_readwrite("joint_limit_activation_distance", &TorqueSafetyParams::joint_limit_activation_distance)
      .def_readwrite("joint_limit_stiffness", &TorqueSafetyParams::joint_limit_stiffness)
      .def_readwrite("joint_limit_damping", &TorqueSafetyParams::joint_limit_damping)
      .def_readwrite("joint_limit_max_torque", &TorqueSafetyParams::joint_limit_max_torque);

  py::class_<FrictionCompensationParams>(m, "FrictionCompensationParams")
      .def(
          py::init<>([](const std::array<double, 7> &coulomb,
                        const std::array<double, 7> &viscous,
                        const std::array<double, 7> &max_torque,
                        double velocity_epsilon) {
            auto friction = FrictionCompensationParams{};
            friction.coulomb = toEigenD<7>(coulomb);
            friction.viscous = toEigenD<7>(viscous);
            friction.max_torque = toEigenD<7>(max_torque);
            friction.velocity_epsilon = velocity_epsilon;
            validateFrictionCompensationParams(friction);
            return friction;
          }),
          "coulomb"_a = std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          "viscous"_a = std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          "max_torque"_a = std::array<double, 7>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
          "velocity_epsilon"_a = 0.03)
      .def_readwrite("coulomb", &FrictionCompensationParams::coulomb)
      .def_readwrite("viscous", &FrictionCompensationParams::viscous)
      .def_readwrite("max_torque", &FrictionCompensationParams::max_torque)
      .def_readwrite("velocity_epsilon", &FrictionCompensationParams::velocity_epsilon);

  py::class_<JointImpedanceParams>(m, "JointImpedanceParams")
      .def(py::init<>())
      .def_readwrite("stiffness", &JointImpedanceParams::stiffness)
      .def_readwrite("damping", &JointImpedanceParams::damping)
      .def_readwrite("constant_torque_offset", &JointImpedanceParams::constant_torque_offset)
      .def_readwrite("compensate_coriolis", &JointImpedanceParams::compensate_coriolis)
      .def_readwrite("safety", &JointImpedanceParams::safety)
      .def_readwrite("friction", &JointImpedanceParams::friction)
      .def_readwrite("cartesian_gains", &JointImpedanceParams::cartesian_gains);

  py::class_<CartesianImpedanceBase::Params>(m, "CartesianImpedanceParams")
      .def(py::init<>())
      .def_readwrite("stiffness", &CartesianImpedanceBase::Params::stiffness)
      .def_readwrite("damping", &CartesianImpedanceBase::Params::damping)
      .def_readwrite("translational_error_clip", &CartesianImpedanceBase::Params::translational_error_clip)
      .def_readwrite("rotational_error_clip", &CartesianImpedanceBase::Params::rotational_error_clip)
      .def_readwrite("force_constraints", &CartesianImpedanceBase::Params::force_constraints)
      .def_readwrite("dynamics_mode", &CartesianImpedanceBase::Params::dynamics_mode)
      .def_readwrite("nullspace_tasks", &CartesianImpedanceBase::Params::nullspace_tasks)
      .def_readwrite("safety", &CartesianImpedanceBase::Params::safety)
      .def_readwrite("friction", &CartesianImpedanceBase::Params::friction);

  py::class_<ExponentialImpedanceMotion::Params, CartesianImpedanceBase::Params>(m, "ExponentialImpedanceParams")
      .def(py::init<>())
      .def_readwrite("target_type", &ExponentialImpedanceMotion::Params::target_type)
      .def_readwrite("exponential_decay", &ExponentialImpedanceMotion::Params::exponential_decay);

  py::class_<CartesianImpedanceMotion::Params, CartesianImpedanceBase::Params>(m, "CartesianImpedanceMotionParams")
      .def(py::init<>())
      .def_readwrite("target_type", &CartesianImpedanceMotion::Params::target_type)
      .def_readwrite("return_when_finished", &CartesianImpedanceMotion::Params::return_when_finished)
      .def_readwrite("finish_wait_factor", &CartesianImpedanceMotion::Params::finish_wait_factor);

  py::class_<JointImpedanceMotion, Motion<franka::Torques>, std::shared_ptr<JointImpedanceMotion>>(
      m, "JointImpedanceMotion")
      .def(
          py::init<>([](const Vector7d &target,
                        std::optional<Vector7d>
                            target_velocity,
                        std::optional<Vector7d>
                            stiffness,
                        std::optional<Vector7d>
                            damping,
                        std::optional<Vector7d>
                            constant_torque_offset,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        bool compensate_coriolis,
                        double max_delta_tau,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        std::optional<FrictionCompensationParams>
                            friction,
                        std::optional<Vector6d>
                            cartesian_stiffness,
                        std::optional<Vector6d>
                            cartesian_damping) {
            auto params = makeJointImpedanceParams(
                stiffness,
                damping,
                constant_torque_offset,
                lower_joint_limits,
                upper_joint_limits,
                compensate_coriolis,
                max_delta_tau,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque,
                friction,
                cartesian_stiffness,
                cartesian_damping);

            const Vector7d target_vector = target;
            if (target_velocity.has_value()) {
              return std::make_shared<JointImpedanceMotion>(target_vector, target_velocity.value(), params);
            }
            return std::make_shared<JointImpedanceMotion>(target_vector, params);
          }),
          R"doc(Construct a static joint impedance controller.

If cartesian_stiffness is provided, the controller additionally shapes its effective
joint stiffness/damping using Cartesian-space gains projected through the current
Jacobian: tau = (diag(stiffness) + J^T diag(cartesian_stiffness) J) @ (q_des - q) + ...,
allowing per-axis (x, y, z, rx, ry, rz) compliance at the end effector while still
tracking a joint-space target. cartesian_damping defaults to critical damping,
2*sqrt(cartesian_stiffness), per axis.)doc",
          "target"_a,
          "target_velocity"_a = std::nullopt,
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "constant_torque_offset"_a = std::nullopt,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "compensate_coriolis"_a = true,
          "max_delta_tau"_a = 1.0,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "friction"_a = std::nullopt,
          "cartesian_stiffness"_a = std::nullopt,
          "cartesian_damping"_a = std::nullopt)
      .def_property_readonly("target", &JointImpedanceMotion::target)
      .def_property_readonly("target_velocity", &JointImpedanceMotion::target_velocity)
      .def_property_readonly("params", [](const JointImpedanceMotion &m) { return m.params(); });

  py::class_<JointImpedanceTrackingMotion, Motion<franka::Torques>, std::shared_ptr<JointImpedanceTrackingMotion>>(
      m, "JointImpedanceTrackingMotion")
      .def(
          py::init<>([](const std::shared_ptr<JointReferenceHandle> &reference_handle,
                        std::optional<Vector7d>
                            stiffness,
                        std::optional<Vector7d>
                            damping,
                        std::optional<Vector7d>
                            constant_torque_offset,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        bool compensate_coriolis,
                        double max_delta_tau,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        std::optional<FrictionCompensationParams>
                            friction,
                        std::optional<Vector6d>
                            cartesian_stiffness,
                        std::optional<Vector6d>
                            cartesian_damping,
                        std::shared_ptr<JointImpedanceGainsHandle>
                            gains_handle,
                        std::shared_ptr<HybridCartesianGainsHandle>
                            cartesian_gains_handle,
                        double gains_time_constant) {
            if (cartesian_gains_handle && !cartesian_stiffness.has_value()) {
              throw py::value_error("cartesian_gains_handle requires cartesian_stiffness to be set");
            }
            auto params = makeJointImpedanceParams(
                stiffness,
                damping,
                constant_torque_offset,
                lower_joint_limits,
                upper_joint_limits,
                compensate_coriolis,
                max_delta_tau,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque,
                friction,
                cartesian_stiffness,
                cartesian_damping);
            if (gains_handle || cartesian_gains_handle) {
              auto runtime = JointImpedanceBase::RuntimeOptions{};
              runtime.gains_handle = std::move(gains_handle);
              runtime.cartesian_gains_handle = std::move(cartesian_gains_handle);
              runtime.gains_time_constant = gains_time_constant;
              return std::make_shared<JointImpedanceTrackingMotion>(reference_handle, params, std::move(runtime));
            }
            return std::make_shared<JointImpedanceTrackingMotion>(reference_handle, params);
          }),
          R"doc(Construct a dynamic joint impedance controller driven by a JointReferenceHandle.

Any constant_torque_offset configured here is added to the per-cycle torque_feedforward values published through the handle.

If gains_handle is provided, the controller reads target gains from it each cycle and exponentially
interpolates toward them with the given time constant, allowing smooth runtime stiffness/damping changes.

If cartesian_stiffness is provided, the controller additionally shapes its effective joint stiffness/damping using
Cartesian-space gains projected through the current Jacobian (see JointImpedanceMotion). If cartesian_gains_handle is
also provided, it is used to update those Cartesian gains at runtime the same way gains_handle updates
stiffness/damping.)doc",
          "reference_handle"_a,
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "constant_torque_offset"_a = std::nullopt,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "compensate_coriolis"_a = true,
          "max_delta_tau"_a = 1.0,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "friction"_a = std::nullopt,
          "cartesian_stiffness"_a = std::nullopt,
          "cartesian_damping"_a = std::nullopt,
          "gains_handle"_a = nullptr,
          "cartesian_gains_handle"_a = nullptr,
          "gains_time_constant"_a = 0.1)
      .def_property_readonly("target", &JointImpedanceTrackingMotion::target)
      .def_property_readonly("target_velocity", &JointImpedanceTrackingMotion::target_velocity)
      .def_property_readonly("params", [](const JointImpedanceTrackingMotion &m) { return m.params(); });

  py::class_<ExponentialImpedanceMotion, CartesianImpedanceBase, std::shared_ptr<ExponentialImpedanceMotion>>(
      m, "ExponentialImpedanceMotion")
      .def(
          py::init<>([](const Affine &target,
                        ReferenceType target_type,
                        std::optional<CartesianGainInput>
                            stiffness,
                        std::optional<CartesianGainInput>
                            damping,
                        std::optional<double>
                            translational_stiffness,
                        std::optional<double>
                            rotational_stiffness,
                        std::optional<double>
                            translational_damping,
                        std::optional<double>
                            rotational_damping,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        double max_delta_tau,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        const Eigen::Vector3d &translational_error_clip,
                        const Eigen::Vector3d &rotational_error_clip,
                        double exponential_decay,
                        const py::object &nullspace_tasks,
                        std::optional<FrictionCompensationParams>
                            friction) {
            auto base_params = makeCartesianImpedanceParams(
                stiffness,
                damping,
                translational_stiffness,
                rotational_stiffness,
                translational_damping,
                rotational_damping,
                force_constraints,
                max_delta_tau,
                lower_joint_limits,
                upper_joint_limits,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque,
                translational_error_clip,
                rotational_error_clip,
                toNullspaceTasks(nullspace_tasks),
                friction);
            auto params = ExponentialImpedanceMotion::Params{};
            static_cast<CartesianImpedanceBase::Params &>(params) = base_params;
            params.target_type = target_type;
            params.exponential_decay = exponential_decay;
            return std::make_shared<ExponentialImpedanceMotion>(target, params);
          }),
          R"doc(Construct an exponential Cartesian impedance motion toward a fixed target pose.

Provide either `stiffness` (a 6x6 matrix or 6-vector, base frame, [x, y, z, rx, ry, rz] order) or the
`translational_stiffness`/`rotational_stiffness` scalars (isotropic, no axis coupling), not both.
Damping defaults to None (critical damping, generalizing 2*sqrt(stiffness) via the matrix square root).
Set explicitly to override.)doc",
          "target"_a,
          py::arg_v("target_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "translational_stiffness"_a = std::nullopt,
          "rotational_stiffness"_a = std::nullopt,
          "translational_damping"_a = std::nullopt,
          "rotational_damping"_a = std::nullopt,
          "force_constraints"_a = std::nullopt,
          "max_delta_tau"_a = 1.0,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "translational_error_clip"_a = Eigen::Vector3d::Constant(0.10),
          "rotational_error_clip"_a = Eigen::Vector3d::Constant(0.25),
          "exponential_decay"_a = 0.005,
          "nullspace_tasks"_a = py::list(),
          "friction"_a = std::nullopt)
      .def_property_readonly("target", &ExponentialImpedanceMotion::target)
      .def_property_readonly("params", [](const ExponentialImpedanceMotion &m) { return m.params(); });

  py::class_<CartesianImpedanceMotion, CartesianImpedanceBase, std::shared_ptr<CartesianImpedanceMotion>>(
      m, "CartesianImpedanceMotion")
      .def(
          py::init<>([](const Affine &target,
                        franka::Duration duration,
                        ReferenceType target_type,
                        std::optional<CartesianGainInput>
                            stiffness,
                        std::optional<CartesianGainInput>
                            damping,
                        std::optional<double>
                            translational_stiffness,
                        std::optional<double>
                            rotational_stiffness,
                        std::optional<double>
                            translational_damping,
                        std::optional<double>
                            rotational_damping,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        double max_delta_tau,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        const Eigen::Vector3d &translational_error_clip,
                        const Eigen::Vector3d &rotational_error_clip,
                        bool return_when_finished,
                        double finish_wait_factor,
                        const py::object &nullspace_tasks,
                        std::optional<FrictionCompensationParams>
                            friction) {
            auto base_params = makeCartesianImpedanceParams(
                stiffness,
                damping,
                translational_stiffness,
                rotational_stiffness,
                translational_damping,
                rotational_damping,
                force_constraints,
                max_delta_tau,
                lower_joint_limits,
                upper_joint_limits,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque,
                translational_error_clip,
                rotational_error_clip,
                toNullspaceTasks(nullspace_tasks),
                friction);
            auto params = CartesianImpedanceMotion::Params{};
            static_cast<CartesianImpedanceBase::Params &>(params) = base_params;
            params.target_type = target_type;
            params.return_when_finished = return_when_finished;
            params.finish_wait_factor = finish_wait_factor;
            return std::make_shared<CartesianImpedanceMotion>(target, duration, params);
          }),
          R"doc(Construct a Cartesian impedance motion that interpolates to a fixed target pose over the given duration.

Provide either `stiffness` (a 6x6 matrix or 6-vector, base frame, [x, y, z, rx, ry, rz] order) or the
`translational_stiffness`/`rotational_stiffness` scalars (isotropic, no axis coupling), not both.
Damping defaults to None (critical damping, generalizing 2*sqrt(stiffness) via the matrix square root).
Set explicitly to override.)doc",
          "target"_a,
          "duration"_a,
          py::arg_v("target_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "translational_stiffness"_a = std::nullopt,
          "rotational_stiffness"_a = std::nullopt,
          "translational_damping"_a = std::nullopt,
          "rotational_damping"_a = std::nullopt,
          "force_constraints"_a = std::nullopt,
          "max_delta_tau"_a = 1.0,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "translational_error_clip"_a = Eigen::Vector3d::Constant(0.10),
          "rotational_error_clip"_a = Eigen::Vector3d::Constant(0.25),
          "return_when_finished"_a = true,
          "finish_wait_factor"_a = 1.2,
          "nullspace_tasks"_a = py::list(),
          "friction"_a = std::nullopt)
      .def_property_readonly("target", &CartesianImpedanceMotion::target)
      .def_property_readonly("duration", &CartesianImpedanceMotion::duration)
      .def_property_readonly("params", [](const CartesianImpedanceMotion &m) { return m.params(); });

  py::class_<
      CartesianImpedanceTrackingMotion,
      CartesianImpedanceBase,
      std::shared_ptr<CartesianImpedanceTrackingMotion>>(m, "CartesianImpedanceTrackingMotion")
      .def(
          py::init<>([](const std::shared_ptr<CartesianReferenceHandle> &reference_handle,
                        std::optional<CartesianGainInput>
                            stiffness,
                        std::optional<CartesianGainInput>
                            damping,
                        std::optional<double>
                            translational_stiffness,
                        std::optional<double>
                            rotational_stiffness,
                        std::optional<double>
                            translational_damping,
                        std::optional<double>
                            rotational_damping,
                        CartesianImpedanceDynamicsMode dynamics_mode,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        double max_delta_tau,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        const Eigen::Vector3d &translational_error_clip,
                        const Eigen::Vector3d &rotational_error_clip,
                        std::shared_ptr<CartesianImpedanceGainsHandle>
                            gains_handle,
                        double gains_time_constant,
                        const py::object &nullspace_tasks,
                        std::optional<FrictionCompensationParams>
                            friction,
                        std::shared_ptr<NullspaceGainsHandle>
                            nullspace_gains_handle) {
            auto base_params = makeCartesianImpedanceParams(
                stiffness,
                damping,
                translational_stiffness,
                rotational_stiffness,
                translational_damping,
                rotational_damping,
                force_constraints,
                max_delta_tau,
                lower_joint_limits,
                upper_joint_limits,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque,
                translational_error_clip,
                rotational_error_clip,
                toNullspaceTasks(nullspace_tasks),
                friction);
            base_params.dynamics_mode = dynamics_mode;
            if (gains_handle || nullspace_gains_handle) {
              auto runtime = CartesianImpedanceBase::RuntimeOptions{};
              runtime.gains_handle = std::move(gains_handle);
              runtime.nullspace_gains_handle = std::move(nullspace_gains_handle);
              runtime.gains_time_constant = gains_time_constant;
              return std::make_shared<CartesianImpedanceTrackingMotion>(
                  reference_handle, base_params, std::move(runtime));
            }
            return std::make_shared<CartesianImpedanceTrackingMotion>(reference_handle, base_params);
          }),
          R"doc(Construct a dynamic Cartesian impedance tracking controller driven by a CartesianReferenceHandle.

Each published Cartesian reference may optionally include a desired end-effector twist in the base frame.
Provide either `stiffness` (a 6x6 matrix or 6-vector, base frame, [x, y, z, rx, ry, rz] order) or the
`translational_stiffness`/`rotational_stiffness` scalars (isotropic, no axis coupling), not both.
Damping defaults to None (critical damping, generalizing 2*sqrt(stiffness) via the matrix square root).
Set explicitly to override.

If gains_handle is provided, the controller reads target gains from it each cycle and exponentially
interpolates toward them with the given time constant, allowing smooth runtime stiffness changes. Gains
read from gains_handle are always full CartesianImpedanceGains matrices (see CartesianImpedanceGains.isotropic
and .diagonal for convenience constructors), independent of how the initial gains above were specified.)doc",
          "reference_handle"_a,
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "translational_stiffness"_a = std::nullopt,
          "rotational_stiffness"_a = std::nullopt,
          "translational_damping"_a = std::nullopt,
          "rotational_damping"_a = std::nullopt,
          py::arg_v(
              "dynamics_mode",
              CartesianImpedanceDynamicsMode::kWrench,
              "_franky.CartesianImpedanceDynamicsMode.Wrench"),
          "force_constraints"_a = std::nullopt,
          "max_delta_tau"_a = 1.0,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "translational_error_clip"_a = Eigen::Vector3d::Constant(0.10),
          "rotational_error_clip"_a = Eigen::Vector3d::Constant(0.25),
          "gains_handle"_a = nullptr,
          "gains_time_constant"_a = 0.1,
          "nullspace_tasks"_a = py::list(),
          "friction"_a = std::nullopt,
          "nullspace_gains_handle"_a = nullptr)
      .def_property_readonly("params", [](const CartesianImpedanceTrackingMotion &m) { return m.params(); });
}
