idf_component_register(
    SRCS "mpu6050.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_driver_i2c freertos driver esp_system esp_hw_support
)