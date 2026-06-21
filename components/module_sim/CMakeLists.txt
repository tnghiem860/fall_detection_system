idf_component_register(
    SRCS "sim_module.c" 
    INCLUDE_DIRS "include"
    REQUIRES esp_driver_uart 
    PRIV_REQUIRES esp_driver_gpio
)
