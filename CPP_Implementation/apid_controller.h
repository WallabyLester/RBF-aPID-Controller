#ifndef APID_CONTROLLER_H
#define APID_CONTROLLER_H

class aPIDController {
public:
    aPIDController(double kp=0.0, double ki=0.0, double kd=0.0, double dt=0.1);

    double update(double target, double measured_value);

    void set_Kp(double kp) {Kp = kp;}
    void set_Ki(double ki) {Ki = ki;}
    void set_Kd(double kd) {Kd = kd;}

    double get_Kp() const { return Kp; }
    double get_Ki() const { return Ki; }
    double get_Kd() const { return Kd; }

private:
    double Kp, Ki, Kd, dt;
    double integral;  
    double prev_err;
};

#endif // APID_CONTROLLER_H
