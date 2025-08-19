#pragma once
#include <cstddef>
#include <initializer_list>
#include <iostream>

#include "CudaMacros.hh"

template <typename T, std::size_t D>
class CudaRall {
public:
  using value_type = T;
  static constexpr std::size_t dimension = D;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  constexpr CudaRall() noexcept = default;
  constexpr CudaRall(CudaRall const &) noexcept = default;
  constexpr CudaRall(CudaRall &&) noexcept = default;
  constexpr CudaRall & operator=(CudaRall const &) noexcept = default;
  constexpr CudaRall & operator=(CudaRall &&) noexcept = default;

  CUDA_HOST CUDA_DEVICE
  explicit constexpr CudaRall(value_type const & value) noexcept : _value(value) {
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = value_type();
    }
  }

  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall & operator=(value_type const & value) noexcept {
    _value = value;
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = value_type();
    }
    return *this;
  }

  CUDA_HOST CUDA_DEVICE
  constexpr CudaRall(value_type const & value, const value_type gradient[dimension]) noexcept : _value(value) {
    for (size_type i = 0; i < dimension; ++i) {
      _gradient[i] = gradient[i];
    }
  }

  CUDA_HOST CUDA_DEVICE
  CudaRall(value_type const & value, std::initializer_list<value_type> gradients) noexcept : _value(value) {
    size_type i = 0;
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

  //
  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator+(CudaRall const & x) noexcept {
    return x;
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

  // bin op

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

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator*(CudaRall const & x, value_type const & y) noexcept {
    CudaRall out(x._value * y);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] * y;
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator*(value_type const & x, CudaRall const & y) noexcept {
    CudaRall out(x * y._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = x * y._gradient[i];
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator/(CudaRall const & x, value_type const & y) noexcept {
    CudaRall out(x._value / y);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = x._gradient[i] / y;
    }
    return out;
  }

  CUDA_HOST CUDA_DEVICE
  friend constexpr CudaRall operator/(value_type const & x, CudaRall const & y) noexcept {
    CudaRall out(x / y._value);
    for (size_type i = 0; i < dimension; ++i) {
      out._gradient[i] = -x * y._gradient[i] / (y._value * y._value);
    }
    return out;
  }

  constexpr value_type const & value() const noexcept { return _value; }
  constexpr value_type const & gradient(size_type i) const noexcept { return _gradient[i]; }
  constexpr value_type const * gradient() const noexcept { return _gradient; }

  friend std::ostream & operator<<(std::ostream & os, CudaRall const & rall) {
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