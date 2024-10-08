#include "apid_controller.h"

aPIDController::aPIDController(double kp, double ki, double kd, double dt)
    : Kp(kp), Ki(ki), Kd(kd), dt(dt), integral(0.0), prev_err(0.0) {}

double aPIDController::update(double target, double measured_value) {
    double error = target - measured_value;
    integral += error * dt;
    double derivative = (error - prev_err) / dt;
    prev_err = error;

    return (Kp * error) + (Ki * integral) + (Kd * derivative);
}
