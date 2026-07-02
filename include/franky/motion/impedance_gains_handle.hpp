#pragma once

#include <Eigen/Eigenvalues>
#include <array>
#include <atomic>
#include <cmath>
#include <optional>

#include "franky/types.hpp"

namespace franky {

inline Vector7d defaultJointImpedanceStiffness() { return Vector7d::Constant(50.0); }

inline Vector7d defaultJointImpedanceDamping(const Vector7d &stiffness) { return 2.0 * stiffness.cwiseSqrt(); }

inline Vector7d defaultJointImpedanceDamping() {
  return defaultJointImpedanceDamping(defaultJointImpedanceStiffness());
}

/**
 * @brief Builds a block-diagonal 6x6 Cartesian gain matrix from isotropic translational and
 * rotational scalars, matching the [x, y, z, rx, ry, rz] axis order of Model::zeroJacobian.
 */
inline Matrix6d cartesianGainBlocks(double translational, double rotational) {
  Matrix6d gains = Matrix6d::Zero();
  gains.topLeftCorner<3, 3>() = translational * Eigen::Matrix3d::Identity();
  gains.bottomRightCorner<3, 3>() = rotational * Eigen::Matrix3d::Identity();
  return gains;
}

inline Matrix6d defaultCartesianImpedanceStiffness() { return cartesianGainBlocks(2000.0, 200.0); }

/**
 * @brief Critical damping for a symmetric PSD Cartesian stiffness matrix, generalizing the
 * scalar rule 2*sqrt(k) via the matrix square root (D = 2*K^(1/2)).
 *
 * For a block-diagonal or diagonal stiffness matrix this reduces exactly to applying 2*sqrt(.)
 * to each eigenvalue/axis independently, matching the elementwise convention used elsewhere
 * (e.g. defaultJointImpedanceDamping).
 */
inline Matrix6d defaultCartesianImpedanceDamping(const Matrix6d &stiffness) {
  Eigen::SelfAdjointEigenSolver<Matrix6d> solver(stiffness);
  return 2.0 * solver.operatorSqrt();
}

/**
 * @brief Target gains for a Cartesian impedance controller.
 *
 * stiffness is a full 6x6 matrix in the base frame, ordered [x, y, z, rx, ry, rz] to match
 * Model::zeroJacobian's row order. It need not be diagonal: off-diagonal terms couple
 * translational and rotational error into the same wrench axis, which is useful e.g. for
 * stiffness expressed about an offset frame. It must be symmetric and positive semi-definite.
 *
 * Damping defaults to nullopt, which means the controller uses critical damping
 * (see defaultCartesianImpedanceDamping). Set explicitly to override.
 */
struct CartesianImpedanceGains {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Matrix6d stiffness{defaultCartesianImpedanceStiffness()};
  std::optional<Matrix6d> damping{std::nullopt};

  /**
   * @brief Convenience factory for the common isotropic case: independent translational and
   * rotational stiffness, with no coupling between axes.
   */
  static CartesianImpedanceGains isotropic(
      double translational_stiffness, double rotational_stiffness,
      std::optional<double> translational_damping = std::nullopt,
      std::optional<double> rotational_damping = std::nullopt) {
    CartesianImpedanceGains gains;
    gains.stiffness = cartesianGainBlocks(translational_stiffness, rotational_stiffness);
    if (translational_damping.has_value() || rotational_damping.has_value()) {
      gains.damping = cartesianGainBlocks(
          translational_damping.value_or(2.0 * std::sqrt(translational_stiffness)),
          rotational_damping.value_or(2.0 * std::sqrt(rotational_stiffness)));
    }
    return gains;
  }

  /**
   * @brief Convenience factory for per-axis stiffness with no cross-axis coupling.
   */
  static CartesianImpedanceGains diagonal(const Vector6d &stiffness, std::optional<Vector6d> damping = std::nullopt) {
    CartesianImpedanceGains gains;
    gains.stiffness = stiffness.asDiagonal();
    if (damping.has_value()) gains.damping = damping->asDiagonal();
    return gains;
  }
};

/**
 * @brief Double-buffered handle for updating Cartesian impedance gains online.
 *
 * Same lock-free pattern as CartesianReferenceHandle. The RT loop reads the
 * target gains each cycle and exponentially interpolates toward them, so
 * stiffness changes are smooth rather than instantaneous.
 *
 * Thread safety: at most one thread may call set() or clear() at a time.
 * Concurrent reads from the RT callback via get() and hasGains() are safe.
 */
class CartesianImpedanceGainsHandle {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CartesianImpedanceGainsHandle() = default;

  void set(const CartesianImpedanceGains &gains);
  void clear();
  [[nodiscard]] bool hasGains() const;
  [[nodiscard]] CartesianImpedanceGains get() const;

 private:
  std::array<CartesianImpedanceGains, 2> buffers_{};
  std::atomic<uint8_t> active_index_{0};
  std::atomic<bool> valid_{false};
};

/**
 * @brief Target gains for a joint impedance controller.
 */
struct JointImpedanceGains {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vector7d stiffness{defaultJointImpedanceStiffness()};
  Vector7d damping{defaultJointImpedanceDamping()};
};

/**
 * @brief Double-buffered handle for updating joint impedance gains online.
 *
 * Thread safety: at most one thread may call set() or clear() at a time.
 * Concurrent reads from the RT callback via get() and hasGains() are safe.
 */
class JointImpedanceGainsHandle {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  JointImpedanceGainsHandle() = default;

  void set(const JointImpedanceGains &gains);
  void clear();
  [[nodiscard]] bool hasGains() const;
  [[nodiscard]] JointImpedanceGains get() const;

 private:
  std::array<JointImpedanceGains, 2> buffers_{};
  std::atomic<uint8_t> active_index_{0};
  std::atomic<bool> valid_{false};
};

}  // namespace franky
