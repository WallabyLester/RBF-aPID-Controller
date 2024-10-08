#include <iostream>
#include "apid_controller.h"

const double Kp = 1.0, Ki = 0.1, Kd = 0.01, dt = 0.1;

int main() {
    aPIDController apid(Kp, Ki, Kd, dt);

    double target = 1.0;
    double measured_value = 0.0;

    double control_signal = apid.update(target, measured_value);

    std::cout << "Output: " << control_signal << std::endl;

    return 0;
}