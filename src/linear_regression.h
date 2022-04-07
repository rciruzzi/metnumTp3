#pragma once

#include "types.h"

class LinearRegression {
public:
    LinearRegression();

    void fit(Matrix A, Vector b);

    Matrix predict(Matrix X);
private:
    Vector solucion;
};
