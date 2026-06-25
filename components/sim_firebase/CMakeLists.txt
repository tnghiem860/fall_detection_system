idf_component_register(
    SRCS "sim_firebase.c"
    INCLUDE_DIRS "include"
    REQUIRES
        module_sim
        driver
        esp_common
)
