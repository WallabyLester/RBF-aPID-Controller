#include <iostream>
#include <cstdlib>
#include <ctime>

#include "rbf_model.h"
#include "apid_controller.h"

// Simple first order system
double simulate_system(double input, double measurement, double dt) {
    return (input - measurement) * dt ;
}

int main() {
    // Seed random number generator for reproducibility
    std::srand(static_cast<unsigned int>(std::time(0)));

    const double Kp = 1.0, Ki = 0.1, Kd = 0.01, dt = 0.1;

    aPIDController apid(Kp, Ki, Kd, dt);

    int n_centers = 5;
    int input_dim = 3;
    double sigma = 1.0;
    RBFModel rbf(n_centers, input_dim, sigma, true);

    double target = 1.0, measured_value = 0.0, learning_rate = 0.01;

    for (int step = 0; step < 100; ++step) {
        double error = target - measured_value;

        double control_signal = apid.update(target, measured_value);

        double gains[3] = {apid.get_Kp(), apid.get_Ki(), apid.get_Kd()};
        rbf.adapt(error, learning_rate, gains);

        control_signal += rbf.predict(gains);
        
        measured_value += simulate_system(control_signal, measured_value, dt);

        std::cout << "Step: " << step 
                << ", Target: " << target
                << ", Measured: " << measured_value
                << ", Error: " << error
                << ", Control Signal: " << control_signal
                << std::endl;
    }
    
    // Example using training on recorded data
    std::cout << "Training Example" << std::endl;
    double inputs[] = {1.0, 0.1, 0.01, 1.0, 0.1, 0.01}; 
    double targets[] = {1.0, 1.0}; 

    rbf.train(inputs, targets, 2, 100, 0.01);

    for (int step = 0; step < 100; ++step) {
        double error = target - measured_value;

        double control_signal = apid.update(target, measured_value);

        double gains[3] = {apid.get_Kp(), apid.get_Ki(), apid.get_Kd()};
        rbf.adapt(error, learning_rate, gains);

        control_signal += rbf.predict(gains);
        
        measured_value += simulate_system(control_signal, measured_value, dt);

        std::cout << "Step: " << step 
                << ", Target: " << target
                << ", Measured: " << measured_value
                << ", Error: " << error
                << ", Control Signal: " << control_signal
                << std::endl;
    }

    return 0;
}