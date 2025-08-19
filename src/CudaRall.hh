#pragma once
#include <cstddef>
#include <initializer_list>
#include <iostream>

#include "CudaMacros.hh"

/// @brief A class for forward-mode automatic differentiation.
/// @tparam T The underlying floating-point type (e.g., float, double).
/// @tparam D The dimension of the gradient vector.
///
/// CudaRall represents a number with both a value and a gradient.
/// It overloads arithmetic operators to propagate derivatives automatically
/// according to the rules of differentiation. It is designed to be usable
/// in both host (CPU) and device (CUDA) code.
template <typename T, std::size_t D>
class CudaRall {
public:
  /// The underlying value type.
  using value_type = T;
  /// The dimension of the gradient vector.
  static constexpr std::size_t dimension = D;
  
  static_assert(T(1) / T(2) != T(0), "CudaRall requires support for division");

  /// @brief Default constructor. Initializes value and gradient to zero.
  constexpr CudaRall() noexcept = default;
  /// @brief Copy constructor.
  constexpr CudaRall(CudaRall const &) noexcept = default;
  /// @brief Move constructor.
  constexpr CudaRall(CudaRall &&) noexcept = default;
  /// @brief Copy assignment operator.
  constexpr CudaRall & operator=(CudaRall const &) noexcept = default;
  /// @brief Move assignment operator.
  constexpr CudaRall & operator=(CudaRall &&) noexcept = default;

  /// @brief Constructs a CudaRall object from a scalar value.
  /// @param value The scalar value. The gradient is initialized to zero.
  CUDA_HOST CUDA_DEVICE
  explicit constexpr CudaRall(value_type const & value) noexcept : _value(value) {
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] = value_type();
    }
  }

  /// @brief Assigns a scalar value to the CudaRall object.
  /// @param value The scalar value. The gradient is reset to zero.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator=(value_type const & value) noexcept {
    _value = value;
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] = value_type();
    }
    return *this;
  }

  /// @brief Constructs a CudaRall object from a value and a gradient array.
  /// @param value The scalar value.
  /// @param gradient A C-style array of size `dimension` for the gradient.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall(value_type const & value, const value_type gradient[dimension]) noexcept : _value(value) {
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] = gradient[i];
    }
  }

  /// @brief Constructs a CudaRall object from a value and an initializer list for the gradient.
  /// @param value The scalar value.
  /// @param gradients An initializer list for the gradient. If the list is shorter than `dimension`, the remaining elements are zero-initialized.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall(value_type const & value, std::initializer_list<value_type> gradients) noexcept : _value(value) {
    std::size_t i = 0;
    for (auto const & grad : gradients) {
      if (i < dimension) {
        _gradient[i] = grad;
        ++i;
      } else {
        break;
      }
    }
    for (; i < dimension; ++i) {
      _gradient[i] = value_type();
    }
  }

  /// @brief Constructs a CudaRall object representing an independent variable.
  /// @param value The scalar value.
  /// @param index The index of the independent variable (0 to dimension-1).
  /// @param gradient The gradient value at the specified index, typically 1.0.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall(value_type const & value, std::size_t index, const value_type& gradient) noexcept : _value(value) {
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] = (i == index) ? gradient : value_type();
    }
  }
  
  /// @brief Templated copy constructor for converting between CudaRall types.
  /// @tparam U The underlying value type of the other CudaRall object.
  /// @param other The CudaRall object to convert from.
  template <typename U>
  CUDA_HOST CUDA_DEVICE
  constexpr explicit CudaRall(CudaRall<U, dimension> const & other) noexcept : _value(other._value) {
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] = other._gradient[i];
    }
  }

  CUDA_HOST CUDA_DEVICE
  void reset() noexcept {
    _value = value_type();
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] = value_type();
    }
  }

  ~CudaRall() = default;

  /// @brief Checks for equality between two CudaRall objects.
  /// @param other The object to compare against.
  /// @return True if both value and gradient are equal, false otherwise.
  CUDA_HOST CUDA_DEVICE
  constexpr bool operator==(CudaRall const& other) const noexcept {
    if (_value != other._value) return false;
    for (std::size_t i = 0; i < dimension; ++i) {
      if (_gradient[i] != other._gradient[i]) return false;
    }
    return true;
  }

  /// @brief Checks for inequality between two CudaRall objects.
  /// @param other The object to compare against.
  /// @return True if either value or gradient are not equal, false otherwise.
  CUDA_HOST CUDA_DEVICE
  constexpr bool operator!=(CudaRall const & other) const noexcept {
    return !(*this == other);
  }

  /// @brief Unary plus operator.
  /// @param x The CudaRall object.
  /// @return A copy of the object.
  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator+(CudaRall const & x) noexcept {
    return x;
  }

  /// @brief Unary negation operator.
  /// @param x The CudaRall object to negate.
  /// @return A new CudaRall object with negated value and gradient.
  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator-(CudaRall const & x) noexcept {
    CudaRall out(-x._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = -x._gradient[i];
    }
    return out;
  }

  /// @brief Compound addition operator.
  /// @param other The CudaRall object to add.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator+=(CudaRall const & other) noexcept {
    _value += other._value;
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] += other._gradient[i];
    }
    return *this;
  }

  /// @brief Compound addition operator for scalar values.
  /// @param other The scalar value to add.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator+=(value_type const & other) noexcept {
    _value += other;
    return *this;
  }

  /// @brief Compound subtraction operator.
  /// @param other The CudaRall object to subtract.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator-=(CudaRall const & other) noexcept {
    _value -= other._value;
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] -= other._gradient[i];
    }
    return *this;
  }

  /// @brief Compound subtraction operator for scalar values.
  /// @param other The scalar value to subtract.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator-=(value_type const & other) noexcept {
    _value -= other;
    return *this;
  }

  /// @brief Compound multiplication operator.
  /// @param other The CudaRall object to multiply by.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator*=(CudaRall const & other) noexcept {
    value_type temp_value = _value;
    _value *= other._value;
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] = _gradient[i] * other._value + temp_value * other._gradient[i];
    }
    return *this;
  }

  /// @brief Compound multiplication operator for scalar values.
  /// @param other The scalar value to multiply by.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator*=(value_type const & other) noexcept {
    _value *= other;
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] *= other;
    }
    return *this;
  }

  /// @brief Compound division operator.
  /// @param other The CudaRall object to divide by.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator/=(CudaRall const & other) noexcept {
    value_type temp_value = _value;
    _value /= other._value;
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] = (_gradient[i] * other._value - temp_value * other._gradient[i]) / (other._value * other._value);
    }
    return *this;
  }

  /// @brief Compound division operator for scalar values.
  /// @param other The scalar value to divide by.
  /// @return A reference to the modified object.
  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator/=(value_type const & other) noexcept {
    _value /= other;
    for (std::size_t i = 0; i < dimension; ++i) {
      _gradient[i] /= other;
    }
    return *this;
  }

  /// @brief Binary addition operator.
  /// @param x The left-hand side operand.
  /// @param y The right-hand side operand.
  /// @return The sum of x and y.
  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator+(CudaRall const & x, CudaRall const & y) noexcept {
    CudaRall out(x._value + y._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] + y._gradient[i];
    }
    return out;
  }

  /// @brief Binary subtraction operator.
  /// @param x The left-hand side operand.
  /// @param y The right-hand side operand.
  /// @return The difference of x and y.
  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator-(CudaRall const & x, CudaRall const & y) noexcept {
    CudaRall out(x._value - y._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] - y._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator*(CudaRall const & x, CudaRall const & y) noexcept {
    CudaRall out(x._value * y._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] * y._value + x._value * y._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator/(CudaRall const & x, CudaRall const & y) noexcept {
    CudaRall out(x._value / y._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = (x._gradient[i] * y._value - x._value * y._gradient[i]) / (y._value * y._value);
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator+(CudaRall const & x, value_type const & y) noexcept {
    CudaRall out(x._value + y);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator+(value_type const & x, CudaRall const & y) noexcept {
    CudaRall out(x + y._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = y._gradient[i];
    }
    return out;
  }
  
  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator-(CudaRall const & x, value_type const & y) noexcept {
    CudaRall out(x._value - y);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator-(value_type const & x, CudaRall const & y) noexcept {
    CudaRall out(x - y._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = -y._gradient[i];
    }
    return out;
  }

  /// @brief Binary multiplication operator.
  /// @param x The left-hand side operand.
  /// @param y The right-hand side operand.
  /// @return The product of x and y.
  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator*(CudaRall const & x, value_type const & y) noexcept {
    CudaRall out(x._value * y);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] * y;
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator*(value_type const & x, CudaRall const & y) noexcept {
    CudaRall out(x * y._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = x * y._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator/(CudaRall const & x, value_type const & y) noexcept {
    CudaRall out(x._value / y);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] / y;
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator/(value_type const & x, CudaRall const & y) noexcept {
    CudaRall out(x / y._value);
    for (std::size_t i = 0; i < dimension; ++i) {
      out._gradient[i] = -x * y._gradient[i] / (y._value * y._value);
    }
    return out;
  }

  constexpr value_type const & value() const noexcept { return _value; }
  constexpr value_type const & gradient(std::size_t i) const noexcept { return _gradient[i]; }
  constexpr value_type const * gradient() const noexcept { return _gradient; }

  constexpr value_type & value() noexcept { return _value; }
  constexpr value_type & gradient(std::size_t i) noexcept { return _gradient[i]; }
  constexpr value_type * gradient() noexcept { return _gradient; }

  template <typename Ch, typename Tr>
  friend std::basic_ostream<Ch, Tr> & operator<<(std::basic_ostream<Ch, Tr> & os, CudaRall const & rall) {
    os << "{ Value: " << rall._value << ", Gradient: [";
    char const * sep = "";
    for (std::size_t i = 0; i < dimension; ++i) {
      os << sep << rall._gradient[i];
      sep = ", ";
    }
    os << "]}";
    return os;
  }

protected:
  value_type _value = value_type();
  value_type _gradient[D] = {0};
};

template <typename T, std::size_t D>
CUDA_HOST CUDA_DEVICE
constexpr CudaRall<T, D> pow(CudaRall<T, D> const & x, int n) noexcept {
  if (n < 0) {
    return 1 / pow(x, -n);
  } else {
    CudaRall<T, D> result(1);
    CudaRall<T, D> base = x;
    while (n > 0) {
      if (n & 0x1) {
        result *= base;
      }
      base *= base;
      n >>= 1;
    }
    return result;    
  }
}

template <typename T, std::size_t D>
CUDA_HOST CUDA_DEVICE
constexpr CudaRall<T, D> sqrt(CudaRall<T, D> const & x) noexcept {
  CudaRall<T, D> out(sqrt(x.value()));
  T const deriv_factor = T(0.5) / out.value();
  for (std::size_t i = 0; i < D; ++i) {
    out._gradient[i] = deriv_factor * x.gradient(i);
  }
  return out;
}