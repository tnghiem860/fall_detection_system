#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "freertos/semphr.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── UART EG800K ───────────────────────────────────────────────────
#define SIM_UART_PORT       UART_NUM_2
#define SIM_UART_TX_PIN     17
#define SIM_UART_RX_PIN     16
// FIX #6: SIM_UART_EN_PIN dùng làm DTR để điều khiển sleep/wake EG800K
#define SIM_UART_EN_PIN     5
#define SIM_UART_BAUD       115200
#define SIM_UART_BUF_SIZE   2048

// ── Tham số GPS ───────────────────────────────────────────────────
#define SIM_GPS_FIX_RETRY_DELAY_MS  3000
#define SIM_GPS_DEFAULT_RETRIES     20

// ── Struct GPS ────────────────────────────────────────────────────
typedef struct {
    bool   valid;
    double lat;
    double lon;
    double alt;
    double speed;
    char   utc_time[24];
    char   utc_date[24];
} sim_gps_t;

typedef void (*sim_gps_cb_t)(sim_gps_t *gps);

// ── API cơ bản ────────────────────────────────────────────────────
esp_err_t sim_init(void);
esp_err_t sim_check_ready(void);
esp_err_t sim_wait_for_network(void);

int sim_send_at_ex(const char *cmd, char *resp_buf, size_t buf_size,
                   uint32_t timeout_ms, const char *expect);
int sim_send_at(const char *cmd, char *resp_buf, size_t buf_size, uint32_t timeout_ms);

// ── SMS ───────────────────────────────────────────────────────────
esp_err_t sim_send_sms(const char *number, const char *message);

// ── GPS ───────────────────────────────────────────────────────────
esp_err_t sim_gps_init(void);
esp_err_t sim_gps_get_location(sim_gps_t *gps, uint32_t timeout_ms);
esp_err_t sim_gps_get_location_retry(sim_gps_t *gps, int retries, uint32_t timeout_each_ms);

// FIX #4: Callback-based background GPS — non-blocking cho fall_detection_task
esp_err_t sim_gps_start_background(const char *number, sim_gps_cb_t cb);


// sim_module.h — thêm vào phần cuối, trước #ifdef __cplusplus
extern SemaphoreHandle_t sim_uart_mutex;

// Macro tiện dùng: lock/unlock với timeout 5s
#define SIM_UART_LOCK()   xSemaphoreTakeRecursive(sim_uart_mutex, pdMS_TO_TICKS(5000))
#define SIM_UART_UNLOCK() xSemaphoreGiveRecursive(sim_uart_mutex)


#ifdef __cplusplus
}
#endif
