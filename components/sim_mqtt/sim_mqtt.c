#include "sim_mqtt.h"
#include "sim_module.h"  // SIM_UART_LOCK / SIM_UART_UNLOCK macros

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "driver/uart.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "SIM_MQTT";

#define SIM_UART_NUM        UART_NUM_2
#define SIM_UART_BUF_SIZE   2048
#define SIM_APN             "m3-world"
#define MQTT_BROKER         "broker.emqx.io"
#define MQTT_PORT           1883
#define MQTT_CLIENT         "ESP32_EG800K_001"

#define MQTT_DEFAULT_MAX_RETRIES  6

static char rx_buf[SIM_UART_BUF_SIZE];

// volatile: compiler khong cache, doc/ghi tu nhieu task van nhin thay dung
static volatile bool s_mqtt_connected = false;

// ============================================================
// PRIVATE helpers — KHONG goi SIM_UART_LOCK, caller tu bao ve
// ============================================================
static void sim_flush(void)
{
    uart_flush_input(SIM_UART_NUM);
    memset(rx_buf, 0, sizeof(rx_buf));
}

static void sim_send_line(const char *cmd)
{
    ESP_LOGI(TAG, ">> %s", cmd);
    uart_write_bytes(SIM_UART_NUM, cmd, strlen(cmd));
    uart_write_bytes(SIM_UART_NUM, "\r\n", 2);
}

static int sim_read_response(char *buf, size_t buf_size, uint32_t timeout_ms,
                              const char *expect, bool stop_on_ok_error)
{
    if (!buf || buf_size == 0) return 0;

    memset(buf, 0, buf_size);
    int total = 0;
    TickType_t start = xTaskGetTickCount();

    while ((xTaskGetTickCount() - start) * portTICK_PERIOD_MS < timeout_ms) {
        int space = (int)buf_size - total - 1;
        if (space <= 0) break;

        int len = uart_read_bytes(SIM_UART_NUM,
                                  (uint8_t *)(buf + total), space,
                                  pdMS_TO_TICKS(50));
        if (len > 0) {
            total += len;
            buf[total] = '\0';

            if (expect && expect[0] != '\0' && strstr(buf, expect)) break;
            if (stop_on_ok_error &&
                (strstr(buf, "OK") || strstr(buf, "ERROR"))) break;
        }
    }

    ESP_LOGI(TAG, "<< %s", buf[0] ? buf : "TIMEOUT/EMPTY");
    return total;
}

static int sim_send_at_wait(const char *cmd, uint32_t timeout_ms, const char *expect)
{
    sim_flush();
    sim_send_line(cmd);
    return sim_read_response(rx_buf, sizeof(rx_buf), timeout_ms, expect,
                             (expect == NULL || expect[0] == '\0'));
}

static bool sim_at_ok(const char *cmd, uint32_t timeout_ms)
{
    sim_send_at_wait(cmd, timeout_ms, "");
    return strstr(rx_buf, "OK") != NULL;
}

// ============================================================
// PUBLIC: Khoi tao
// FIX: them SIM_UART_LOCK bao ve truy cap UART
// ============================================================
bool sim_mqtt_init(void)
{
    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "[MQTT] init: cannot get UART mutex");
        return false;
    }
    ESP_LOGI(TAG, "=== MQTT init ===");
    vTaskDelay(pdMS_TO_TICKS(500));
    sim_at_ok("ATE0",      3000);
    sim_at_ok("AT+CMEE=2", 3000);
    SIM_UART_UNLOCK();
    return true;
}

// FIX: Release mutex giua cac vong delay de task khac (SMS) co the dung UART
bool sim_mqtt_wait_network(void)
{
    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "[MQTT] wait_network: cannot get UART mutex");
        return false;
    }
    ESP_LOGI(TAG, "Waiting for network registration...");

    for (int i = 0; i < 20; i++) {
        sim_send_at_wait("AT+CREG?",  3000, "");
        bool creg_ok  = (strstr(rx_buf, ",1") != NULL || strstr(rx_buf, ",5") != NULL);

        sim_send_at_wait("AT+CGREG?", 3000, "");
        bool cgreg_ok = (strstr(rx_buf, ",1") != NULL || strstr(rx_buf, ",5") != NULL);

        if (creg_ok && cgreg_ok) {
            ESP_LOGI(TAG, "Network registered OK");
            SIM_UART_UNLOCK();
            return true;
        }

        ESP_LOGW(TAG, "Waiting network %d/20...", i + 1);
        // Tra mutex trong thoi gian cho de task uu tien cao hon (VD: SMS) dung UART
        SIM_UART_UNLOCK();
        vTaskDelay(pdMS_TO_TICKS(3000));
        if (SIM_UART_LOCK() != pdTRUE) return false;
    }

    ESP_LOGE(TAG, "Network registration FAILED");
    SIM_UART_UNLOCK();
    return false;
}

// ============================================================
// PUBLIC: GPRS / PDP
// FIX: Lock/unlock rieng, release mutex truoc cac delay dai
// ============================================================
bool sim_mqtt_start_pdp(void)
{
    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "[GPRS] start_pdp: cannot get UART mutex");
        return false;
    }

    char cmd[160];
    ESP_LOGI(TAG, "[GPRS] Configuring TCP/IP stack...");

    snprintf(cmd, sizeof(cmd), "AT+QICSGP=1,1,\"%s\",\"\",\"\",1", SIM_APN);
    sim_at_ok(cmd, 5000);
    sim_at_ok("AT+QIDEACT=1", 10000);

    // Tra mutex truoc delay dai (2s)
    SIM_UART_UNLOCK();
    vTaskDelay(pdMS_TO_TICKS(2000));
    if (SIM_UART_LOCK() != pdTRUE) return false;

    sim_at_ok("AT+QIACT=1", 30000);

    // Tra mutex truoc delay dai (3s)
    SIM_UART_UNLOCK();
    vTaskDelay(pdMS_TO_TICKS(3000));
    if (SIM_UART_LOCK() != pdTRUE) return false;

    sim_send_at_wait("AT+QIACT?", 5000, "");
    bool ok = strstr(rx_buf, "1,1,1") != NULL;
    SIM_UART_UNLOCK();

    if (ok) {
        ESP_LOGI(TAG, "[GPRS] TCP/IP stack ready");
    } else {
        ESP_LOGE(TAG, "[GPRS] Failed to get IP");
    }
    return ok;
}

// ============================================================
// PUBLIC: MQTT Connect
// FIX: Lock rieng, release truoc delay
// ============================================================
bool sim_mqtt_connect_broker(void)
{
    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "[MQTT] connect_broker: cannot get UART mutex");
        return false;
    }

    char cmd[192];
    ESP_LOGI(TAG, "[MQTT] Opening TCP to broker...");

    sim_send_line("AT+QMTCLOSE=0");
    // Tra mutex truoc delay (2s)
    SIM_UART_UNLOCK();
    vTaskDelay(pdMS_TO_TICKS(2000));
    if (SIM_UART_LOCK() != pdTRUE) return false;
    sim_flush();

    sim_at_ok("AT+QMTCFG=\"version\",0,4",    3000);
    sim_at_ok("AT+QMTCFG=\"keepalive\",0,120", 3000);

    snprintf(cmd, sizeof(cmd), "AT+QMTOPEN=0,\"%s\",%d", MQTT_BROKER, MQTT_PORT);
    sim_send_at_wait(cmd, 30000, "+QMTOPEN:");

    if (!strstr(rx_buf, "+QMTOPEN: 0,0")) {
        ESP_LOGE(TAG, "[MQTT] QMTOPEN failed: %s", rx_buf);
        s_mqtt_connected = false;
        SIM_UART_UNLOCK();
        return false;
    }

    ESP_LOGI(TAG, "[MQTT] TCP opened, sending CONNECT...");
    snprintf(cmd, sizeof(cmd), "AT+QMTCONN=0,\"%s\"", MQTT_CLIENT);
    sim_send_at_wait(cmd, 15000, "+QMTCONN:");

    bool connected = strstr(rx_buf, "+QMTCONN: 0,0,0") != NULL;
    s_mqtt_connected = connected;

    if (connected) {
        ESP_LOGI(TAG, "[MQTT] Connected to broker");
    } else {
        ESP_LOGE(TAG, "[MQTT] Connect failed: %s", rx_buf);
    }
    SIM_UART_UNLOCK();
    return connected;
}

bool sim_mqtt_is_connected(void)
{
    if (SIM_UART_LOCK() != pdTRUE) {
        // Fallback: tra ve trang thai cache neu khong lay duoc mutex
        return s_mqtt_connected;
    }
    sim_send_at_wait("AT+QMTCONN?", 5000, "");
    bool connected = strstr(rx_buf, "+QMTCONN: 0,3") != NULL;
    s_mqtt_connected = connected;
    SIM_UART_UNLOCK();
    return connected;
}

// ============================================================
// PUBLIC: ensure_connection
// FIX: Khong giu mutex xuyen suot — moi buoc UART tu lock/unlock.
// Trong khoang vTaskDelay giua cac retry, mutex duoc giai phong
// -> sos_button_task co the gui SMS khan cap ma khong bi chan.
// ============================================================
bool sim_mqtt_ensure_connection(int max_retries)
{
    // Kiem tra nhanh — is_connected tu lay/tra mutex
    if (sim_mqtt_is_connected()) return true;

    if (max_retries <= 0) max_retries = MQTT_DEFAULT_MAX_RETRIES;

    ESP_LOGI(TAG, "[MQTT] Reconnecting (max %d attempts)...", max_retries);

    int gprs_reinit_done = 0;

    for (int retry = 1; retry <= max_retries; retry++) {
        ESP_LOGI(TAG, "[MQTT] Attempt %d/%d", retry, max_retries);

        // Khi qua nua so lan thu, reinit GPRS mot lan
        if (!gprs_reinit_done && retry > max_retries / 2) {
            ESP_LOGW(TAG, "[MQTT] Reinit GPRS...");
            sim_mqtt_start_pdp();  // tu lay/tra mutex
            gprs_reinit_done = 1;
            vTaskDelay(pdMS_TO_TICKS(3000));  // khong giu mutex trong delay
        }

        if (sim_mqtt_connect_broker()) return true;  // tu lay/tra mutex

        vTaskDelay(pdMS_TO_TICKS(5000));  // khong giu mutex trong delay
    }

    ESP_LOGE(TAG, "[MQTT] Failed after %d attempts", max_retries);
    return false;
}

// ============================================================
// PUBLIC: Publish
// (da co lock tu truoc, giu nguyen)
// ============================================================
bool sim_mqtt_publish(const char *topic, const char *payload)
{
    if (!topic || !payload) return false;
    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "[MQTT] publish: cannot get UART mutex");
        return false;
    }

    char cmd[192];
    snprintf(cmd, sizeof(cmd), "AT+QMTPUB=0,0,0,0,\"%s\"", topic);

    sim_flush();
    sim_send_line(cmd);

    sim_read_response(rx_buf, sizeof(rx_buf), 8000, ">", false);
    if (!strstr(rx_buf, ">")) {
        ESP_LOGE(TAG, "[MQTT] No publish prompt '>'");
        SIM_UART_UNLOCK();
        return false;
    }

    uart_write_bytes(SIM_UART_NUM, payload, strlen(payload));
    uint8_t ctrl_z = 0x1A;
    uart_write_bytes(SIM_UART_NUM, (const char *)&ctrl_z, 1);

    ESP_LOGI(TAG, "[MQTT] Topic: %s", topic);
    ESP_LOGI(TAG, "[MQTT] Payload: %s", payload);

    sim_read_response(rx_buf, sizeof(rx_buf), 10000, "+QMTPUB:", false);

    bool ok = strstr(rx_buf, "+QMTPUB: 0,0,0") != NULL;
    if (ok) {
        ESP_LOGI(TAG, "[MQTT] Publish OK");
    } else {
        ESP_LOGE(TAG, "[MQTT] Publish FAILED: %s", rx_buf);
    }
    SIM_UART_UNLOCK();
    return ok;
}

// ============================================================
// PUBLIC: Wrappers
// ============================================================
bool sim_mqtt_publish_location(const char *device_id, float lat, float lon,
                               const char *source)
{
    (void)device_id;
    (void)source;

    char payload[160];
    snprintf(payload, sizeof(payload),
             "{\"lat\":%.6f,\"lon\":%.6f}", lat, lon);

    return sim_mqtt_publish(MQTT_LOCATION_TOPIC, payload);
}

bool sim_mqtt_publish_fall_alert(const char *device_id, float confidence,
                                 float lat, float lon, bool has_location)
{
    char payload[300];

    if (has_location) {
        snprintf(payload, sizeof(payload),
                 "{\"device_id\":\"%s\",\"fall\":true,\"confidence\":%.2f"
                 ",\"lat\":%.6f,\"lon\":%.6f"
                 ",\"maps\":\"https://maps.google.com/?q=%.6f,%.6f\"}",
                 device_id ? device_id : MQTT_CLIENT,
                 confidence, lat, lon, lat, lon);
    } else {
        snprintf(payload, sizeof(payload),
                 "{\"device_id\":\"%s\",\"fall\":true,\"confidence\":%.2f"
                 ",\"lat\":null,\"lon\":null,\"maps\":null}",
                 device_id ? device_id : MQTT_CLIENT,
                 confidence);
    }

    return sim_mqtt_publish(MQTT_ALERT_TOPIC, payload);
}

bool sim_mqtt_publish_status(const char *device_id, bool is_fall, float confidence)
{
    char payload[160];
    snprintf(payload, sizeof(payload),
             "{\"device_id\":\"%s\",\"fall\":%s,\"confidence\":%.2f}",
             device_id ? device_id : MQTT_CLIENT,
             is_fall ? "true" : "false",
             confidence);

    return sim_mqtt_publish(MQTT_STATUS_TOPIC, payload);
}

bool sim_mqtt_disconnect(void)
{
    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "[MQTT] disconnect: cannot get UART mutex");
        return false;
    }
    s_mqtt_connected = false;
    sim_at_ok("AT+QMTDISC=0", 5000);
    sim_at_ok("AT+QMTCLOSE=0", 5000);
    SIM_UART_UNLOCK();
    return true;
}
