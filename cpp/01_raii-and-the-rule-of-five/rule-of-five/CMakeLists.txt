add_executable(1_rule-of-three 1_rule-of-three.cpp
        point-impl.h)
target_link_libraries(1_rule-of-three GTest::gtest GTest::gtest_main GTest::gmock)

add_executable(2_lvalue-vs-rvalue 2_lvalue-vs-rvalue.cpp)

add_executable(3_rule-of-five 3_rule-of-five.cpp)
target_link_libraries(3_rule-of-five GTest::gtest GTest::gtest_main GTest::gmock)
#add_executable(value-ref-and-pointer value-ref-and-pointer.cpp)
