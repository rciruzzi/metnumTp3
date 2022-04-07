#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;

LinearRegression::LinearRegression(){
}

void LinearRegression::fit(Matrix A, Vector b) {
    Matrix aux = (A.transpose() * A);
    Vector auxB = A.transpose() * b;
    solucion =  aux.fullPivLu().solve(auxB);
}

Matrix LinearRegression::predict(Matrix X) {
    return X * solucion;
}
