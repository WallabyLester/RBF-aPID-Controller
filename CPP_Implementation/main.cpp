#include <iostream>
#include <cstdlib>
#include <ctime>

#include "rbf_model.h"
#include "apid_controller.h"

// Simple first order system
double systemResponse(double input, double measurement, double dt) {
    return (input - measurement) * dt ;
}

int main() {
    // Seed random number generator for reproducibility
    std::srand(static_cast<unsigned int>(std::time(0)));
    
    const double Kp = 1.0, Ki = 0.1, Kd = 0.01, dt = 0.1;

    aPIDController apid(Kp, Ki, Kd, dt);

    double target = 1.0;
    double measured_value = 0.0;

    double control_signal = apid.update(target, measured_value);

    std::cout << "Output: " << control_signal << std::endl;

    return 0;
}