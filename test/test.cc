#include "CudaRall.hh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

int main()
{
  // test CudaRall class

  CudaRall<double, 3> a(2.0, {3.0, 4.0, 5.0});
  CudaRall<double, 3> b(3.0, {6.0, 7.0, 8.0});
  CudaRall<double, 3> c = a + b;
  CudaRall<double, 3> d = a - b;
  CudaRall<double, 3> e = a * b;
  CudaRall<double, 3> f = a / b;

  std::cout << "a = " << a << std::endl;
  std::cout << "b = " << b << std::endl;
  std::cout << "c = " << c << std::endl;
  std::cout << "d = " << d << std::endl;
  std::cout << "e = " << e << std::endl;
  std::cout << "f = " << f << std::endl;
  assert((c == CudaRall<double, 3>(5.0, {9.0, 11.0, 13.0})));
  assert((d == CudaRall<double, 3>(-1.0, {-3.0, -3.0, -3.0})));
  assert((e == CudaRall<double, 3>(6.0, {21.0, 26.0, 31.0})));



  std::cout << a * a + pow(a, 5) << std::endl;


  return 0;
}