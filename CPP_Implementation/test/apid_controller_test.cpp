#include <gtest/gtest.h>
#include "apid_controller.h"

// Test fixture for aPIDController
class aPIDControllerTest : public ::testing::Test {
protected:
    aPIDController* apid;

    void SetUp() override {
        apid = new aPIDController(1.0, 0.1, 0.01, 0.1);
    }

    void TearDown() override {
        delete apid; 
    }
};

// Test case for control signal with zero error
TEST_F(aPIDControllerTest, Control_Signal_With_Zero_Error) {
    double target = 1.0;
    double measured_value = 1.0;

    double control_signal = apid->update(target, measured_value);
    EXPECT_NEAR(control_signal, 0.0, 1e-5); 
}

// Test case for control signal with positive error
TEST_F(aPIDControllerTest, Control_Signal_With_Positive_Error) {
    double target = 10.0;
    double measured_value = 5.0;

    double control_signal = apid->update(target, measured_value);
    EXPECT_GT(control_signal, 0.0);
}

// Test case for control signal with negative error
TEST_F(aPIDControllerTest, Control_Signal_With_Negative_Error) {
    double target = 5.0;
    double measured_value = 10.0;

    double control_signal = apid->update(target, measured_value);
    EXPECT_LT(control_signal, 0.0);
}

// Test case for proportional action
TEST_F(aPIDControllerTest, Proportional_Action) {
    double Kp = 2.0;
    apid->set_Kp(Kp);
    apid->set_Ki(0.0);
    apid->set_Kd(0.0);
    double target = 10.0;
    
    double measured_value = 5.0;
    double control_signal_pos = apid->update(target, measured_value);
    EXPECT_NEAR(control_signal_pos, Kp * (target - measured_value), 1e-5); 

    measured_value = 15.0;
    double control_signal_neg = apid->update(target, measured_value);
    EXPECT_NEAR(control_signal_neg, Kp * (target - measured_value), 1e-5);
}

// Test case for integral action
TEST_F(aPIDControllerTest, Integral_Action) {
    double Kp = 4.0, Ki = 2.0;
    apid->set_Kp(Kp);
    apid->set_Ki(Ki);
    double target = 10.0;
    double measured_value = 0.0;
    double dt = 0.1;

    for (int i = 0; i < 10; ++i) {
        double control_signal = apid->update(target, measured_value);
        measured_value += (control_signal - measured_value) * dt; 
    }

    EXPECT_NEAR(measured_value, target, 1.0); 
}

// Test case for derivative action
TEST_F(aPIDControllerTest, Derivative_Action) {
    double Kd = 1.0;
    apid->set_Kp(0.0);
    apid->set_Ki(0.0);
    apid->set_Kd(Kd);
    double target = 10.0;

    double measured_value = 5.0; 
    apid->update(target, measured_value);

    measured_value = 15.0; 
    double controlSignal = apid->update(target, measured_value);

    double expectedDerivative = (target - measured_value)/0.1 - (target - 5.0)/0.1; 
    EXPECT_NEAR(controlSignal, Kd * expectedDerivative, 1e-5); 
}
