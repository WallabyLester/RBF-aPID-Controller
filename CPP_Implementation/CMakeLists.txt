cmake_minimum_required(VERSION 3.18)

project(RBF_aPID
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(src)

set(SOURCE_FILES
    src/apid_controller.cpp
    src/rbf_model.cpp
)

set(TEST_FILES
    test/apid_controller_test.cpp
    test/rbf_model_test.cpp    
)

add_library(ModelLibrary ${SOURCE_FILES})

add_executable(control_system src/main.cpp)
target_link_libraries(control_system ModelLibrary)

add_executable(model_tests ${TEST_FILES})
target_link_libraries(model_tests GTest::GTest GTest::Main ModelLibrary)

add_test(NAME RunAllTests COMMAND model_tests)
