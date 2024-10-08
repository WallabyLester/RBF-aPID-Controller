#include <iostream>
#include <cmath>



class aPIDController {
public:
    aPIDController(double kp, double ki, double kd, double dt)
        : Kp(kp), Ki(ki), Kd(kd), dt(dt), integral(0.0), prev_err(0.0) {}

    double update(double target, double measured_value) {
        double error = target - measured_value;
        integral += error * dt;
        double derivative = (error - prev_err) / dt;
        prev_err = error;

        return (Kp * error) + (Ki * integral) + (Kd * derivative);
    }

private:
    double Kp, Ki, Kd, dt;
    double integral;
    double prev_err;
};

int main() {
    // PID parameters
    double Kp = 1.0, Ki = 0.1, Kd = 0.01, dt = 0.1;
    
    aPIDController apid(Kp, Ki, Kd, dt);

    double target = 1.0, measured_value = 0.0;

    double control_signal = apid.update(target, measured_value);

    std::cout << "Output: " << control_signal << std::endl;
    
    return 0;
}
