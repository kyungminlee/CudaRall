#pragma once
#include <cstddef>
#include "CudaMacros.hh"

template <typename T, std::size_t D>
class CudaRall {
public:
  using value_type = T;
  static constexpr std::size_t dimension = D;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  friend class CudaRall;

  constexpr CudaRall() = default;
  constexpr CudaRall(CudaRall const &) = default;
  constexpr CudaRall(CudaRall &&) = default;
  constexpr CudaRall & operator=(CudaRall const &) = default;
  constexpr CudaRall & operator=(CudaRall &&) = default;

  CUDA_HOST CUDA_DEVICE
  explicit constexpr CudaRall(value_type const & value) noexcept : _value(value) {
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = value_type();
    }
  }

  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall(value_type const & value, const value_type gradient[dimension]) noexcept : _value(value) {
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = gradient[i];
    }
  }

  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall(value_type const & value, size_type index, const value_type& gradient) noexcept : _value(value) {
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = (i == index) ? gradient : value_type();
    }
  }
  
  template <typename U>
  CUDA_HOST CUDA_DEVICE
  constexpr explicit CudaRall(CudaRall<U, dimension> const & other) noexcept : _value(other._value) {
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = other._gradient[i];
    }
  }

  CUDA_HOST CUDA_DEVICE
  constexpr bool operator==(CudaRall const& other) const noexcept {
    if (_value != other._value) return false;
    for (size_type i = 0; i < dimension; ++i) {
      if (_gradient[i] != other._gradient[i]) return false;
    }
    return true;
  }

  CUDA_HOST CUDA_DEVICE
  constexpr bool operator!=(CudaRall const & other) const noexcept {
    return !(*this == other);
  }

  CUDA_HOST CUDA_DEVICE
  constexpr bool operator<(CudaRall const & other) const noexcept {
    if (_value != other._value) return _value < other._value;
    for (size_type i = 0; i < dimension; ++i) {
      if (_gradient[i] != other._gradient[i]) return _gradient[i] < other._gradient[i];
    }
    return false;
  }

  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator+=(CudaRall const & other) noexcept {
    _value += other._value;
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] += other._gradient[i];
    }
    return *this;
  }

  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator-=(CudaRall const & other) noexcept {
    _value -= other._value;
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] -= other._gradient[i];
    }
    return *this;
  }

  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator*=(CudaRall const & other) noexcept {
    value_type temp_value = _value;
    _value *= other._value;
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = _gradient[i] * other._value + temp_value * other._gradient[i];
    }
    return *this;
  }

  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator/=(CudaRall const & other) noexcept {
    value_type temp_value = _value;
    _value /= other._value;
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = (_gradient[i] * other._value - temp_value * other._gradient[i]) / (other._value * other._value);
    }
    return *this;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator+(CudaRall const & x, CudaRall const & y) noexcept {
    CudaRall out(x._value + y._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] + y._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator-(CudaRall const & x, CudaRall const & y) noexcept {
    CudaRall out(x._value - y._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] - y._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator*(CudaRall const & x, CudaRall const & y) noexcept {
    CudaRall out(x._value * y._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] * y._value + x._value * y._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator/(CudaRall const & x, CudaRall const & y) noexcept {
    CudaRall out(x._value / y._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = (x._gradient[i] * y._value - x._value * y._gradient[i]) / (y._value * y._value);
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator-(CudaRall const & x) noexcept {
    CudaRall out(-x._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = -x._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator+(CudaRall const & x, value_type const & y) noexcept {
    CudaRall out(x._value + y);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator+(value_type const & x, CudaRall const & y) noexcept {
    CudaRall out(x + y._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = y._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator-(CudaRall const & x, value_type const & y) noexcept {
    CudaRall out(x._value - y);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator-(value_type const & x, CudaRall const & y) noexcept {
    CudaRall out(x - y._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = -y._gradient[i];
    }
    return out;
  }

protected:
  value_type _value = value_type();
  value_type _gradient[D] = {0};
};

template <typename T, std::size_t D>
CudaRall<T, D> pow(CudaRall<T, D> const & x, std::size_t n) {
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
