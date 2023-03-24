set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

add_definitions(-DCROW_ENABLE_SSL)
add_executable(cs main.cpp)
add_library(deviceManager deviceManager.cpp)


target_link_libraries(cs deviceManager pthread rt ssl crypto spdlog::spdlog)

add_custom_command(
  TARGET cs POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cs> ../../
)
