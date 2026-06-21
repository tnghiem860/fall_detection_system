#pragma once

#include <stdint.h>
#include "esp_err.h"
#include "driver/i2c_master.h"   // API mới v5.x — không dùng driver/i2c.h

// ============================================================
// Cấu hình I2C
// ============================================================
#define I2C_MASTER_SCL_IO       22
#define I2C_MASTER_SDA_IO       21
#define I2C_MASTER_NUM          I2C_NUM_0
#define I2C_MASTER_FREQ_HZ      100000

#define MPU6050_ADDR            0x68
#define MPU6050_INT_GPIO        GPIO_NUM_34

// ============================================================
// Thanh ghi
// ============================================================
#define MPU6050_PWR_MGMT_1      0x6B
#define MPU6050_ACCEL_XOUT_H    0x3B
#define MPU6050_REG_SMPLRT_DIV  0x19
#define MPU6050_REG_CONFIG      0x1A
#define MPU6050_REG_ACCEL_CFG   0x1C
#define MPU6050_REG_GYRO_CFG    0x1B
#define MPU6050_REG_WHO_AM_I    0x75
#define MPU6050_REG_FIFO_EN     0x23
#define MPU6050_REG_USER_CTRL   0x6A
#define MPU6050_REG_FIFO_COUNT_H 0x72
#define MPU6050_REG_FIFO_R_W    0x74
#define MPU6050_REG_INT_EN      0x38   // Interrupt Enable
#define MPU6050_REG_INT_CFG     0x37   // Interrupt Config  
#define MPU6050_REG_INT_STATUS  0x3A   // Interrupt Status

// ============================================================
// FIX #1: Scale nhất quán với cấu hình thanh ghi ±16g
// Register 0x1C = 0x18 → ±16g → LSB = 2048 LSB/g
// Model AI được train với dữ liệu ±16g nên phải dùng giá trị này.
// ============================================================
#define ACCEL_LSB_PER_G          2048.0f   // ±16g
#define GYRO_LSB_PER_DPS        131.0f     // ±250°/s

// Số byte mỗi mẫu FIFO (Accel X,Y,Z = 3 × 2 byte = 6 byte)
#define MPU6050_FIFO_BYTES_PER_SAMPLE  6
// FIFO overflow interrupt bit
#define MPU6050_INT_FIFO_OFLOW  (1 << 4)
// Data ready interrupt bit  
#define MPU6050_INT_DATA_RDY    (1 << 0)
// ============================================================
// Struct dữ liệu
// ============================================================
typedef struct {
    float ax_g;
    float ay_g;
    float az_g;
    float gx_dps;
    float gy_dps;
    float gz_dps;
    float accel_magnitude;
} mpu6050_data_t;

// ============================================================
// API
// ============================================================
esp_err_t i2c_master_init(i2c_master_bus_handle_t *out_bus_handle);
esp_err_t mpu6050_init(i2c_master_bus_handle_t bus_handle);
esp_err_t mpu6050_read(mpu6050_data_t *data);

// FIX #5 (workflow): Đọc batch từ FIFO — cần thiết khi dùng Light Sleep thực sự
// Trả về số mẫu đọc được (có thể 0 nếu FIFO chưa đủ)
int mpu6050_read_fifo(mpu6050_data_t *samples, int max_samples);
// mpu6050.h — thêm vào phần API
esp_err_t mpu6050_clear_interrupt(void);
