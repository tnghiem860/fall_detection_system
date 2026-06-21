#include "mpu6050.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>
#include <string.h>
#include "driver/gpio.h"
#include "esp_sleep.h"

static const char *TAG = "MPU6050";

static i2c_master_dev_handle_t s_dev_handle = NULL;

// ============================================================
// Private helpers
// ============================================================
static esp_err_t mpu6050_write_byte(uint8_t reg_addr, uint8_t data)
{
    uint8_t write_buf[2] = {reg_addr, data};
    return i2c_master_transmit(s_dev_handle, write_buf, sizeof(write_buf),
                               pdMS_TO_TICKS(1000));
}

static esp_err_t mpu6050_read_bytes(uint8_t reg_addr, uint8_t *data, size_t len)
{
    return i2c_master_transmit_receive(s_dev_handle, &reg_addr, 1, data, len,
                                       pdMS_TO_TICKS(1000));
}

// ============================================================
// i2c_master_init — v5.x API
// ============================================================
esp_err_t i2c_master_init(i2c_master_bus_handle_t *out_bus_handle)
{
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port      = I2C_MASTER_NUM,
        .sda_io_num    = I2C_MASTER_SDA_IO,
        .scl_io_num    = I2C_MASTER_SCL_IO,
        .clk_source    = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };

    esp_err_t err = i2c_new_master_bus(&bus_cfg, out_bus_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2c_new_master_bus failed: %s", esp_err_to_name(err));
        return err;
    }

    ESP_LOGI(TAG, "I2C master initialized (v5.x API)");
    return ESP_OK;
}

// ============================================================
// mpu6050_init
// FIX #1: Thanh ghi ACCEL_CFG = 0x00 → ±2g (LSB = 16384)
//         Nhất quán với ACCEL_LSB_PER_G trong header.
// FIX #5: Bật FIFO cho Accel — cần cho Light Sleep batch read.
// ============================================================
esp_err_t mpu6050_init(i2c_master_bus_handle_t bus_handle)
{
    i2c_device_config_t dev_cfg = {
        .device_address  = MPU6050_ADDR,
        .scl_speed_hz    = I2C_MASTER_FREQ_HZ,
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
    };

    esp_err_t err = i2c_master_bus_add_device(bus_handle, &dev_cfg, &s_dev_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2c_master_bus_add_device failed: %s", esp_err_to_name(err));
        return err;
    }

    // Wake up, dùng PLL với X gyro
    err = mpu6050_write_byte(MPU6050_PWR_MGMT_1, 0x01);
    if (err != ESP_OK) return err;
    vTaskDelay(pdMS_TO_TICKS(100));

    // Sample rate 25Hz
    err = mpu6050_write_byte(MPU6050_REG_SMPLRT_DIV, 39);
    if (err != ESP_OK) return err;

    // DLPF ~21Hz
    err = mpu6050_write_byte(MPU6050_REG_CONFIG, 0x04);
    if (err != ESP_OK) return err;

    // ±16g
    err = mpu6050_write_byte(MPU6050_REG_ACCEL_CFG, 0x18);
    if (err != ESP_OK) return err;

    // ±250°/s
    err = mpu6050_write_byte(MPU6050_REG_GYRO_CFG, 0x00);
    if (err != ESP_OK) return err;

    // Bật FIFO cho Accel (bit 3)
   /* err = mpu6050_write_byte(MPU6050_REG_FIFO_EN, 0x08);
    if (err != ESP_OK) return err;

    // Reset FIFO trước, rồi bật
    err = mpu6050_write_byte(MPU6050_REG_USER_CTRL, 0x44);
    if (err != ESP_OK) return err;
    vTaskDelay(pdMS_TO_TICKS(10));
    err = mpu6050_write_byte(MPU6050_REG_USER_CTRL, 0x40);
    if (err != ESP_OK) return err;*/

    // SỬA MỚI: Cấu hình INT pin — active high, push-pull, clear on any read
    // 0x00 = active high | push-pull | latch disabled | clear on status read
    err = mpu6050_write_byte(MPU6050_REG_INT_CFG, 0x00);
    if (err != ESP_OK) return err;

    // SỬA MỚI: Bật interrupt FIFO Overflow (bit 4)
    // Khi FIFO gần đầy, MPU kéo INT pin HIGH → đánh thức ESP32
    err = mpu6050_write_byte(MPU6050_REG_INT_EN, MPU6050_INT_DATA_RDY);
    if (err != ESP_OK) return err;

    // SỬA MỚI: Cấu hình GPIO34 nhận INT từ MPU6050
    gpio_config_t int_cfg = {
        .pin_bit_mask = 1ULL << MPU6050_INT_GPIO,
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,  // MPU kéo lên nội bộ
        .pull_down_en = GPIO_PULLDOWN_DISABLE,  // Giữ LOW khi không có INT
        .intr_type    = GPIO_INTR_DISABLE,     // Không dùng ISR, chỉ làm wakeup source
    };
    gpio_config(&int_cfg);

    // Cho phép GPIO34 đánh thức ESP32 từ Light Sleep (level HIGH)
    esp_sleep_enable_ext0_wakeup(MPU6050_INT_GPIO, 1);

    ESP_LOGI(TAG, "MPU6050 init: ±16g, 25Hz, FIFO+INT enabled on GPIO%d",
             MPU6050_INT_GPIO);
    uint8_t check;
mpu6050_read_bytes(MPU6050_REG_ACCEL_CFG, &check, 1);
ESP_LOGI(TAG, "ACCEL_CFG register = 0x%02X (expect 0x18)", check);

uint8_t who;
mpu6050_read_bytes(MPU6050_REG_WHO_AM_I, &who, 1);
ESP_LOGI(TAG, "WHO_AM_I = 0x%02X (expect 0x68)", who);
    return ESP_OK;
}

// Thêm hàm clear interrupt status sau khi đọc FIFO
// Cần gọi sau mpu6050_read_fifo() để MPU hạ INT pin xuống LOW
esp_err_t mpu6050_clear_interrupt(void)
{
    uint8_t status;
    // Đọc INT_STATUS register là đủ để clear (nếu INT_CFG = 0x00)
    return mpu6050_read_bytes(MPU6050_REG_INT_STATUS, &status, 1);
}

// ============================================================
// mpu6050_read — đọc trực tiếp từ thanh ghi (dùng khi không sleep)
// ============================================================
esp_err_t mpu6050_read(mpu6050_data_t *data)
{
    uint8_t buf[14];

    esp_err_t err = mpu6050_read_bytes(MPU6050_ACCEL_XOUT_H, buf, 14);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read MPU6050: %s", esp_err_to_name(err));
        return err;
    }

    int16_t ax_raw = (int16_t)((buf[0]  << 8) | buf[1]);
    int16_t ay_raw = (int16_t)((buf[2]  << 8) | buf[3]);
    int16_t az_raw = (int16_t)((buf[4]  << 8) | buf[5]);
    // buf[6..7] = nhiệt độ, bỏ qua
    int16_t gx_raw = (int16_t)((buf[8]  << 8) | buf[9]);
    int16_t gy_raw = (int16_t)((buf[10] << 8) | buf[11]);
    int16_t gz_raw = (int16_t)((buf[12] << 8) | buf[13]);

    data->ax_g   = ax_raw / ACCEL_LSB_PER_G;
    data->ay_g   = ay_raw / ACCEL_LSB_PER_G;
    data->az_g   = az_raw / ACCEL_LSB_PER_G;
    data->gx_dps = gx_raw / GYRO_LSB_PER_DPS;
    data->gy_dps = gy_raw / GYRO_LSB_PER_DPS;
    data->gz_dps = gz_raw / GYRO_LSB_PER_DPS;

    data->accel_magnitude = sqrtf(
        data->ax_g * data->ax_g +
        data->ay_g * data->ay_g +
        data->az_g * data->az_g
    );

    return ESP_OK;
}

// ============================================================
// FIX #5: mpu6050_read_fifo
// Đọc toàn bộ accel samples từ FIFO buffer của MPU6050.
// Dùng khi ESP32 thức dậy từ Light Sleep — lấy hết batch đã tích lũy.
// Chỉ đọc FIFO Accel (6 byte/mẫu: AX_H, AX_L, AY_H, AY_L, AZ_H, AZ_L).
// Trả về số mẫu thực sự đọc được (0 nếu FIFO rỗng hay lỗi).
// ============================================================
int mpu6050_read_fifo(mpu6050_data_t *samples, int max_samples)
{
    if (!samples || max_samples <= 0) return 0;

    // Đọc FIFO_COUNT (2 byte)
    uint8_t cnt_buf[2];
    if (mpu6050_read_bytes(MPU6050_REG_FIFO_COUNT_H, cnt_buf, 2) != ESP_OK) return 0;

    int fifo_count = (int)((cnt_buf[0] << 8) | cnt_buf[1]);
    if (fifo_count <= 0) return 0;

    // Mỗi mẫu Accel FIFO = 6 byte (X,Y,Z × 2 byte)
    int num_samples = fifo_count / MPU6050_FIFO_BYTES_PER_SAMPLE;
    if (num_samples > max_samples) num_samples = max_samples;

    int read_count = 0;
    for (int i = 0; i < num_samples; i++) {
        uint8_t buf[6];
        if (mpu6050_read_bytes(MPU6050_REG_FIFO_R_W, buf, 6) != ESP_OK) break;

        int16_t ax_raw = (int16_t)((buf[0] << 8) | buf[1]);
        int16_t ay_raw = (int16_t)((buf[2] << 8) | buf[3]);
        int16_t az_raw = (int16_t)((buf[4] << 8) | buf[5]);

        samples[i].ax_g   = ax_raw / ACCEL_LSB_PER_G;
        samples[i].ay_g   = ay_raw / ACCEL_LSB_PER_G;
        samples[i].az_g   = az_raw / ACCEL_LSB_PER_G;
        samples[i].gx_dps = 0.0f;  // Gyro không trong FIFO mode này
        samples[i].gy_dps = 0.0f;
        samples[i].gz_dps = 0.0f;
        samples[i].accel_magnitude = sqrtf(
            samples[i].ax_g * samples[i].ax_g +
            samples[i].ay_g * samples[i].ay_g +
            samples[i].az_g * samples[i].az_g
        );
        read_count++;
    }

    return read_count;
}
