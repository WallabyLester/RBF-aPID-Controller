#ifndef APID_CONTROLLER_H
#define APID_CONTROLLER_H

class aPIDController {
public:
    aPIDController(double kp, double ki, double kd, double dt);

    double update(double target, double measured_value);

private:
    double Kp, Ki, Kd, dt;
    double integral;  
    double prev_err;
};

#endif // APID_CONTROLLER_H
