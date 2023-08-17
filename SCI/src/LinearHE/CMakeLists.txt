find_package(OpenMP REQUIRED)

find_package(SEAL 4.0.0 EXACT QUIET PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
if (NOT SEAL_FOUND)
    message(STATUS "SEAL 4.0.0 was not found: clone and install SEAL locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/SEAL/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/SEAL")
        execute_process(COMMAND git submodule update --init --recursive extern/SEAL
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        #execute_process(COMMAND git apply "${PROJECT_SOURCE_DIR}/cmake/seal.patch"
            #WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    execute_process(COMMAND ${CMAKE_COMMAND} -DSEAL_USE_INTEL_HEXL=ON
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build .
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    find_package(SEAL 4.0.0 EXACT REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
endif()

add_library(SCI-LinearHE
    conv-field.cpp
    fc-field.cpp
    elemwise-prod-field.cpp
    utils-HE.cpp
    utils-vec.cpp
)

target_link_libraries(SCI-LinearHE
    PUBLIC
    SCI-utils
    SEAL::seal
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)
