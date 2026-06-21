idf_component_register(
    SRCS "sim_mqtt.c"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_timer module_sim
)