idf_component_register(
    SRCS "main.cc"
    INCLUDE_DIRS "."
    REQUIRES
        ai_model
        module_sim
        mpu6050
        sim_firebase
)