#ifndef APID_CONTROLLER_H
#define APID_CONTROLLER_H

/**
 * @class aPIDController
 * @brief Adaptive PID Controller class for control systems.
 * 
 * This class implements a simple adaptive PID controller that can be used
 * to control a system by adjusting the output based on the error
 * between a target value and a measured value.
 */
class aPIDController {
public:
    /**
     * @brief Constructor to initialize PID gains and time step.
     * 
     * @param kp Proportional gain.
     * @param ki Integral gain.
     * @param kd Derivative gain.
     * @param dt Time step.
     */
    aPIDController(double kp=0.0, double ki=0.0, double kd=0.0, double dt=0.1);

    /**
     * @brief Update the PID output based on the target and measured value.
     * 
     * @param target The desired target value.
     * @param measured_value The current measured value.
     * @return The computed control output.
     */
    double update(double target, double measured_value);

    /** 
     * @brief Set the proportional gain.
     * @param kp The new proportional gain.
     */
    void set_Kp(double kp) {Kp = kp;}

    /** 
     * @brief Set the integral gain.
     * @param ki The new integral gain.
     */
    void set_Ki(double ki) {Ki = ki;}

    /** 
     * @brief Set the derivative gain.
     * @param kd The new derivative gain.
     */
    void set_Kd(double kd) {Kd = kd;}

    /**
     * @brief Get the proportional gain.
     * @return The current proportional gain.
     */
    double get_Kp() const { return Kp; }

    /**
     * @brief Get the integral gain.
     * @return The current integral gain.
     */
    double get_Ki() const { return Ki; }

    /**
     * @brief Get the derivative gain.
     * @return The current derivative gain.
     */
    double get_Kd() const { return Kd; }

private:
    double Kp, Ki, Kd;  // PID gains
    double dt;          // Time step
    double integral;    // Integral term
    double prev_err;    // Previous error
};

#endif // APID_CONTROLLER_H
