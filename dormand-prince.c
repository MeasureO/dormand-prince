#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

const double a = 1.0;
// Define the system of differential equations
double f1(double t, double x1, double x2) {
    return x2;
}

double f2(double t, double x1, double x2) {
    return -x1 * (a + t * t * t) / t ;
}

// Define the fourth-order Runge-Kutta method
void rk4(double *t, double *x1, double *x2, double h, double (*f1)(double, double, double), double (*f2)(double, double, double)) {
    double k1x1 = h * (*f1)(*t, *x1, *x2);
    double k1x2 = h * (*f2)(*t, *x1, *x2);
    double k2x1 = h * (*f1)(*t + h/2.0, *x1 + k1x1/2.0, *x2 + k1x2/2.0);
    double k2x2 = h * (*f2)(*t + h/2.0, *x1 + k1x1/2.0, *x2 + k1x2/2.0);
    double k3x1 = h * (*f1)(*t + h/2.0, *x1 + k2x1/2.0, *x2 + k2x2/2.0);
    double k3x2 = h * (*f2)(*t + h/2.0, *x1 + k2x1/2.0, *x2 + k2x2/2.0);
    double k4x1 = h * (*f1)(*t + h, *x1 + k3x1, *x2 + k3x2);
    double k4x2 = h * (*f2)(*t + h, *x1 + k3x1, *x2 + k3x2);

    *x1 += (k1x1 + 2.0*k2x1 + 2.0*k3x1 + k4x1)/6.0;
    *x2 += (k1x2 + 2.0*k2x2 + 2.0*k3x2 + k4x2)/6.0;
    *t += h;
}

// Define the adaptive step-size Runge-Kutta method
double rk4_auto(double *t, double *x1, double *x2, double h, double h_min, double h_max, double tolerance, double safety_factor, bool error_control, double (*f1)(double, double, double), double (*f2)(double, double, double)) {
    double x1_old = *x1;
    double x2_old = *x2;
    double error_ratio = 0.0;
    double h_new = 0.0;
    int iter = 0;

    while (iter < 100) {
        // Take two steps with step size h
        rk4(t, x1, x2, h, f1, f2);
        rk4(t, x1, x2, h, f1, f2);

        // Take one step with step size h/2
        double x1_half = x1_old;
        double x2_half = x2_old;
        double t_half = *t - h;
        rk4(&t_half, &x1_half, &x2_half, h/2.0, f1, f2);

        // Calculate the error
        double error = fabs(x1_half - *x1) / 15.0;

        // Update the step size
        if (error_control) {
            error_ratio = tolerance / fabs(error);
            if (error_ratio >= 1.0) {
                h_new = safety_factor * h * pow(error_ratio, 0.25);
            } else {
                h_new = safety_factor * h * pow(error_ratio, 0.2);
            }
        } else {
            h_new = safety_factor * h * pow(tolerance / error, 0.25);
        }
        double h_old = h;
        h = fmax(h_min, fmin(h_max, h_new));

        // Check if the new step size is too small
        if (fabs(h - h_old) < 1e-10) {
            printf("Error: step size too small\n");
            exit(1);
        }

        // Check if the error is small enough
        if (error_control && error_ratio < 1.0) {
            x1_old = *x1;
            x2_old = *x2;
        } else {
            return error;
        }

        iter++;
    }

    printf("Error: maximum number of iterations reached\n");
    exit(1);
}

int main() {
    double t = 0.05;
    double x1 = 1.0;
    double x2 = 9.0;
    double h = 0.1;
    double h_min = 1e-6;
    double h_max = 1.0;
    double tolerance = 1e-6;
    double safety_factor = 0.9;
    bool error_control = true;

    double t_end = 1.0;

    while (t < t_end) {
        double error = rk4_auto(&t, &x1, &x2, h, h_min, h_max, tolerance, safety_factor, error_control, f1, f2);
        printf("%.6f %.6f %.6f %.6f\n", t, x1, x2, error);
    }

    return 0;
}
