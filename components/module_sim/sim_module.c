#include "sim_module.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "SIM_MODEM";
SemaphoreHandle_t sim_uart_mutex = NULL;
static bool s_uart_installed = false;
static bool s_gps_started    = false;
static TaskHandle_t s_gps_task_handle = NULL;

// FIX #3: Spinlock bao ve check+create cua GPS task -- tranh race condition
// khi ca fall_detection_task va sos_button_task cung goi start_background.
static portMUX_TYPE     s_gps_mux         = portMUX_INITIALIZER_UNLOCKED;
static volatile bool    s_gps_task_running = false;

typedef struct {
    char number[24];
    sim_gps_cb_t cb;
} gps_task_arg_t;

static void sim_flush_input(void)
{
    uart_flush_input(SIM_UART_PORT);
}

int sim_send_at_ex(const char *cmd, char *resp_buf, size_t buf_size,
                   uint32_t timeout_ms, const char *expect)
{
    if (!cmd || !resp_buf || buf_size == 0) return -1;

    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "sim_send_at_ex: không lấy được mutex (timeout)");
        return -1;
    }

    sim_flush_input();
    memset(resp_buf, 0, buf_size);

    ESP_LOGI(TAG, ">> %s", cmd);
    uart_write_bytes(SIM_UART_PORT, cmd, strlen(cmd));
    uart_write_bytes(SIM_UART_PORT, "\r\n", 2);

    int total = 0;
    TickType_t start = xTaskGetTickCount();

    while ((xTaskGetTickCount() - start) * portTICK_PERIOD_MS < timeout_ms) {
        int space_left = (int)buf_size - total - 1;
        if (space_left <= 0) break;

        int len = uart_read_bytes(SIM_UART_PORT,
                                  (uint8_t *)(resp_buf + total),
                                  space_left,
                                  pdMS_TO_TICKS(50));
        if (len > 0) {
            total += len;
            resp_buf[total] = '\0';

            if (expect && strstr(resp_buf, expect)) break;
            if (!expect && (strstr(resp_buf, "OK") || strstr(resp_buf, "ERROR"))) break;
            if (strstr(resp_buf, "+CME ERROR") || strstr(resp_buf, "+CMS ERROR")) break;
        }
    }

    ESP_LOGI(TAG, "<< %s", resp_buf[0] ? resp_buf : "TIMEOUT/EMPTY");
    SIM_UART_UNLOCK();
    return total;
}

int sim_send_at(const char *cmd, char *resp_buf, size_t buf_size, uint32_t timeout_ms)
{
    return sim_send_at_ex(cmd, resp_buf, buf_size, timeout_ms, NULL);
}

esp_err_t sim_init(void)
{
    if (sim_uart_mutex == NULL) {
        sim_uart_mutex = xSemaphoreCreateRecursiveMutex();
        configASSERT(sim_uart_mutex);
    }

    if (s_uart_installed) return ESP_OK;

    gpio_config_t io_conf = {
        .pin_bit_mask  = (1ULL << SIM_UART_EN_PIN),
        .mode          = GPIO_MODE_OUTPUT,
        .pull_up_en    = GPIO_PULLUP_DISABLE,
        .pull_down_en  = GPIO_PULLDOWN_DISABLE,
        .intr_type     = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    // FIX #6: Khởi động với EN = HIGH (modem active)
    gpio_set_level((gpio_num_t)SIM_UART_EN_PIN, 1);

    uart_config_t uart_cfg = {
        .baud_rate           = SIM_UART_BAUD,
        .data_bits           = UART_DATA_8_BITS,
        .parity              = UART_PARITY_DISABLE,
        .stop_bits           = UART_STOP_BITS_1,
        .flow_ctrl           = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk          = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_param_config(SIM_UART_PORT, &uart_cfg));
    ESP_ERROR_CHECK(uart_set_pin(SIM_UART_PORT,
                                 SIM_UART_TX_PIN, SIM_UART_RX_PIN,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    esp_err_t err = uart_driver_install(SIM_UART_PORT,
                                        SIM_UART_BUF_SIZE, SIM_UART_BUF_SIZE,
                                        0, NULL, 0);
    if (err == ESP_ERR_INVALID_STATE) {
        s_uart_installed = true;
        return ESP_OK;
    }
    if (err != ESP_OK) return err;

    s_uart_installed = true;
    vTaskDelay(pdMS_TO_TICKS(3000));
    ESP_LOGI(TAG, "UART SIM ready: UART%d, TX=%d, RX=%d, baud=%d",
             SIM_UART_PORT, SIM_UART_TX_PIN, SIM_UART_RX_PIN, SIM_UART_BAUD);
    return ESP_OK;
}

esp_err_t sim_check_ready(void)
{
    char buf[256];

    for (int i = 0; i < 10; i++) {
        sim_send_at("AT", buf, sizeof(buf), 1000);
        if (strstr(buf, "OK")) break;
        vTaskDelay(pdMS_TO_TICKS(1000));
        if (i == 9) return ESP_FAIL;
    }

    sim_send_at("ATE0",      buf, sizeof(buf), 2000);
    sim_send_at("AT+CMEE=2", buf, sizeof(buf), 2000);
    sim_send_at("ATI",       buf, sizeof(buf), 3000);
    sim_send_at("AT+CPIN?",  buf, sizeof(buf), 3000);
    if (!strstr(buf, "+CPIN: READY")) {
        ESP_LOGE(TAG, "SIM chưa READY");
        return ESP_FAIL;
    }

    sim_send_at("AT+CSQ", buf, sizeof(buf), 3000);
    return sim_wait_for_network();
}

esp_err_t sim_wait_for_network(void)
{
    char creg[128];
    char cgreg[128];

    ESP_LOGI(TAG, "Chờ đăng ký mạng...");
    for (int i = 0; i < 20; i++) {
        sim_send_at("AT+CREG?",  creg,  sizeof(creg),  2000);
        sim_send_at("AT+CGREG?", cgreg, sizeof(cgreg), 2000);

        bool reg_ok  = strstr(creg,  ",1") || strstr(creg,  ",5");
        bool gprs_ok = strstr(cgreg, ",1") || strstr(cgreg, ",5");

        if (reg_ok && gprs_ok) {
            ESP_LOGI(TAG, "Đã đăng ký mạng");
            return ESP_OK;
        }

        ESP_LOGW(TAG, "Chưa đăng ký mạng, chờ 3s...");
        vTaskDelay(pdMS_TO_TICKS(3000));
    }

    return ESP_FAIL;
}

esp_err_t sim_send_sms(const char *number, const char *message)
{
    if (!number || !message) return ESP_ERR_INVALID_ARG;

    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "sim_send_sms: không lấy được mutex");
        return ESP_FAIL;
    }

    char buf[256];
    char cmd[64];

    sim_send_at("AT+CMGF=1",       buf, sizeof(buf), 3000);
    sim_send_at("AT+CSCS=\"GSM\"", buf, sizeof(buf), 3000);

    snprintf(cmd, sizeof(cmd), "AT+CMGS=\"%s\"", number);
    sim_send_at_ex(cmd, buf, sizeof(buf), 8000, ">");
    if (!strstr(buf, ">")) {
        ESP_LOGE(TAG, "Không nhận được prompt gửi SMS");
        SIM_UART_UNLOCK();
        return ESP_FAIL;
    }

    uart_write_bytes(SIM_UART_PORT, message, strlen(message));
    uint8_t ctrl_z = 0x1A;
    uart_write_bytes(SIM_UART_PORT, (const char *)&ctrl_z, 1);

    memset(buf, 0, sizeof(buf));
    int total = 0;
    TickType_t start = xTaskGetTickCount();
    while ((xTaskGetTickCount() - start) * portTICK_PERIOD_MS < 30000) {
        int len = uart_read_bytes(SIM_UART_PORT,
                                  (uint8_t *)(buf + total),
                                  sizeof(buf) - total - 1,
                                  pdMS_TO_TICKS(100));
        if (len > 0) {
            total += len;
            buf[total] = '\0';
            if (strstr(buf, "+CMGS:") && strstr(buf, "OK")) {
                ESP_LOGI(TAG, "SMS sent: %s", buf);
                SIM_UART_UNLOCK(); 
                return ESP_OK;
            }
            if (strstr(buf, "ERROR") || strstr(buf, "+CMS ERROR")) break;
        }
    }

    ESP_LOGE(TAG, "Gửi SMS thất bại: %s", buf[0] ? buf : "TIMEOUT");
    SIM_UART_UNLOCK();
    return ESP_FAIL;
}

esp_err_t sim_gps_init(void)
{
    char buf[256];

    ESP_LOGI(TAG, "Khởi động GPS engine...");
    sim_send_at("AT+QGPSEND", buf, sizeof(buf), 3000);
    vTaskDelay(pdMS_TO_TICKS(1000));

    sim_send_at("AT+QGPS=1", buf, sizeof(buf), 5000);
    if (strstr(buf, "OK") || strstr(buf, "already") || strstr(buf, "+CME ERROR: 504")) {
        s_gps_started = true;
        ESP_LOGI(TAG, "GPS engine bật, chờ fix ngoài trời");
        return ESP_OK;
    }

    ESP_LOGE(TAG, "Không bật được GPS: %s", buf);
    return ESP_FAIL;
}

static void copy_token(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0) return;
    if (!src) { dst[0] = '\0'; return; }
    snprintf(dst, dst_size, "%s", src);
}

esp_err_t sim_gps_get_location(sim_gps_t *gps, uint32_t timeout_ms)
{
    if (!gps) return ESP_ERR_INVALID_ARG;

    memset(gps, 0, sizeof(*gps));
    gps->valid = false;

    if (!s_gps_started) sim_gps_init();

    char buf[512];
    sim_send_at("AT+QGPSLOC=2", buf, sizeof(buf), timeout_ms);

    char *p = strstr(buf, "+QGPSLOC:");
    if (!p) {
        ESP_LOGW(TAG, "Chưa có GPS fix");
        return ESP_FAIL;
    }

    p += strlen("+QGPSLOC:");
    while (*p == ' ') p++;

    char *line_end = strpbrk(p, "\r\n");
    if (line_end) *line_end = '\0';

    char *tokens[11] = {0};
    int count = 0;
    char *saveptr = NULL;
    char *tok = strtok_r(p, ",", &saveptr);
    while (tok && count < 11) {
        tokens[count++] = tok;
        tok = strtok_r(NULL, ",", &saveptr);
    }

    if (count < 8 || !tokens[1] || !tokens[2]) {
        ESP_LOGE(TAG, "GPS parse lỗi: %s", p);
        return ESP_FAIL;
    }

    gps->valid = true;
    copy_token(gps->utc_time, sizeof(gps->utc_time), tokens[0]);
    gps->lat   = atof(tokens[1]);
    gps->lon   = atof(tokens[2]);
    gps->alt   = tokens[4] ? atof(tokens[4]) : 0.0;
    gps->speed = tokens[7] ? atof(tokens[7]) : 0.0;
    copy_token(gps->utc_date, sizeof(gps->utc_date), (count > 9) ? tokens[9] : "N/A");

    ESP_LOGI(TAG, "GPS Fix: lat=%.6f lon=%.6f alt=%.1f speed=%.1f",
             gps->lat, gps->lon, gps->alt, gps->speed);
    return ESP_OK;
}

esp_err_t sim_gps_get_location_retry(sim_gps_t *gps, int retries, uint32_t timeout_each_ms)
{
    if (retries <= 0) retries = SIM_GPS_DEFAULT_RETRIES;

    for (int i = 0; i < retries; i++) {
        ESP_LOGI(TAG, "Đọc GPS lần %d/%d", i + 1, retries);
        if (sim_gps_get_location(gps, timeout_each_ms) == ESP_OK && gps && gps->valid)
            return ESP_OK;
        vTaskDelay(pdMS_TO_TICKS(SIM_GPS_FIX_RETRY_DELAY_MS));
    }

    return ESP_FAIL;
}

static void gps_background_task(void *arg)
{
    gps_task_arg_t *task_arg = (gps_task_arg_t *)arg;
    sim_gps_t gps;

    // FIX: Gửi SMS cảnh báo NGAY — không chờ GPS
    if (task_arg && task_arg->number[0]) {
        sim_send_sms(task_arg->number,
                     "CANH BAO TE NGA! Dang xac dinh vi tri...");
    }

    // Sau đó mới thử lấy GPS (chạy song song với SMS đã gửi)
    esp_err_t err = sim_gps_get_location_retry(&gps, SIM_GPS_DEFAULT_RETRIES, 5000);

    if (err == ESP_OK && gps.valid) {
        // Có GPS — gửi SMS thứ 2 với tọa độ
        if (task_arg && task_arg->number[0]) {
            char msg[180];
            snprintf(msg, sizeof(msg),
                     "Vi tri: https://maps.google.com/?q=%.6f,%.6f",
                     gps.lat, gps.lon);
            sim_send_sms(task_arg->number, msg);
        }
        if (task_arg && task_arg->cb) {
            task_arg->cb(&gps);
        }
    } else {
        ESP_LOGW(TAG, "Không lấy được GPS sau nhiều lần thử");
        // SMS đầu đã gửi rồi — chỉ cần callback để publish MQTT fallback
        if (task_arg && task_arg->cb) {
            sim_gps_t empty = {0};
            task_arg->cb(&empty);
        }
    }

    free(task_arg);
    // FIX #3: Xoa handle va reset flag duoi spinlock de tranh race voi start_background
    taskENTER_CRITICAL(&s_gps_mux);
    s_gps_task_handle  = NULL;
    s_gps_task_running = false;
    taskEXIT_CRITICAL(&s_gps_mux);
    vTaskDelete(NULL);
}

esp_err_t sim_gps_start_background(const char *number, sim_gps_cb_t cb)
{
    // FIX #3: Atomically check+reserve slot truoc khi tao task.
    // xTaskCreatePinnedToCore (heap alloc) khong duoc goi trong critical section,
    // nen dung flag s_gps_task_running dat trong CS de dat cho truoc.
    taskENTER_CRITICAL(&s_gps_mux);
    if (s_gps_task_running) {
        taskEXIT_CRITICAL(&s_gps_mux);
        ESP_LOGW(TAG, "GPS task already running, skipping");
        return ESP_ERR_INVALID_STATE;
    }
    s_gps_task_running = true;  // dat cho truoc — tranh task thu 2 di qua check
    taskEXIT_CRITICAL(&s_gps_mux);

    gps_task_arg_t *arg = (gps_task_arg_t *)calloc(1, sizeof(gps_task_arg_t));
    if (!arg) {
        taskENTER_CRITICAL(&s_gps_mux);
        s_gps_task_running = false;
        taskEXIT_CRITICAL(&s_gps_mux);
        return ESP_ERR_NO_MEM;
    }

    if (number) snprintf(arg->number, sizeof(arg->number), "%s", number);
    arg->cb = cb;

    BaseType_t ok = xTaskCreatePinnedToCore(gps_background_task,
                                            "sim_gps_bg", 10240,
                                            arg, 4,
                                            &s_gps_task_handle, 0);
    if (ok != pdPASS) {
        free(arg);
        taskENTER_CRITICAL(&s_gps_mux);
        s_gps_task_handle  = NULL;
        s_gps_task_running = false;
        taskEXIT_CRITICAL(&s_gps_mux);
        return ESP_FAIL;
    }

    return ESP_OK;
}


