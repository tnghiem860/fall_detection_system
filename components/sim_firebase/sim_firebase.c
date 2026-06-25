#include "sim_firebase.h"
#include "sim_module.h"   // SIM_UART_LOCK / SIM_UART_UNLOCK

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "driver/uart.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define SIM_APN "v-internet"

static const char *TAG = "SIM_FIREBASE";

static bool fb_at(const char *cmd, uint32_t timeout_ms, const char *expect);

bool sim_firebase_start_pdp(void)
{
    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "[GPRS] start_pdp: cannot get UART mutex");
        return false;
    }

    char cmd[160];
    ESP_LOGI(TAG, "[GPRS] Configuring TCP/IP stack...");

    snprintf(cmd, sizeof(cmd), "AT+QICSGP=1,1,\"%s\",\"\",\"\",1", SIM_APN);
    fb_at(cmd, 5000, "OK");
    fb_at("AT+QIDEACT=1", 10000, "OK");

    // Tra mutex truoc delay dai
    SIM_UART_UNLOCK();
    vTaskDelay(pdMS_TO_TICKS(2000));
    if (SIM_UART_LOCK() != pdTRUE) return false;

    fb_at("AT+QIACT=1", 30000, "OK");

    // Kiem tra IP
    bool got_ip = false;
    if (fb_at("AT+QIACT?", 5000, "+QIACT: 1")) {
        ESP_LOGI(TAG, "[GPRS] PDP context 1 is active (Got IP)");
        got_ip = true;
    }

    // Config HTTP to use PDP context 1
    fb_at("AT+QHTTPSTOP", 2000, NULL);
    fb_at("AT+QHTTPCFG=\"contextid\",1", 3000, "OK");
    fb_at("AT+QHTTPCFG=\"responseheader\",1", 3000, "OK");

    SIM_UART_UNLOCK();
    return got_ip;
}

#define SIM_UART_NUM      UART_NUM_2
#define RX_BUF_SIZE       2048
#define HTTP_INPUT_TO     10    // giây để gửi body
#define HTTP_RESP_TO      30    // giây chờ response

// URL paths
#define COMMIT_PATH \
    "/v1/projects/" FIREBASE_PROJECT_ID \
    "/databases/(default)/documents:commit?key=" FIREBASE_API_KEY

#define FALL_EVENTS_PATH \
    "/v1/projects/" FIREBASE_PROJECT_ID \
    "/databases/(default)/documents/fall_events?key=" FIREBASE_API_KEY

static char s_rx_buf[RX_BUF_SIZE];

// ── helpers ──────────────────────────────────────────────────────

static void fb_flush(void)
{
    uart_flush_input(SIM_UART_NUM);
    memset(s_rx_buf, 0, sizeof(s_rx_buf));
}

static void fb_send(const char *data)
{
    uart_write_bytes(SIM_UART_NUM, data, strlen(data));
}

static void fb_sendline(const char *cmd)
{
    ESP_LOGI(TAG, ">> %s", cmd);
    uart_write_bytes(SIM_UART_NUM, cmd, strlen(cmd));
    uart_write_bytes(SIM_UART_NUM, "\r\n", 2);
}

static int fb_read(uint32_t timeout_ms, const char *expect)
{
    memset(s_rx_buf, 0, sizeof(s_rx_buf));
    int total = 0;
    TickType_t start = xTaskGetTickCount();

    while ((xTaskGetTickCount() - start) * portTICK_PERIOD_MS < timeout_ms) {
        int space = (int)sizeof(s_rx_buf) - total - 1;
        if (space <= 0) break;
        int n = uart_read_bytes(SIM_UART_NUM, (uint8_t *)(s_rx_buf + total),
                                space, pdMS_TO_TICKS(100));
        if (n > 0) {
            total += n;
            s_rx_buf[total] = '\0';
            if (expect && strstr(s_rx_buf, expect)) break;
            if (strstr(s_rx_buf, "ERROR")) break;
            if (!expect && strstr(s_rx_buf, "OK")) break;
        }
    }
    ESP_LOGI(TAG, "<< %s", total ? s_rx_buf : "TIMEOUT");
    return total;
}

static bool fb_at(const char *cmd, uint32_t timeout_ms, const char *expect)
{
    fb_flush();
    fb_sendline(cmd);
    fb_read(timeout_ms, expect);
    return (expect && strstr(s_rx_buf, expect)) ||
           (!expect && strstr(s_rx_buf, "OK"));
}

// ── Thực hiện HTTP POST lên Firestore ────────────────────────────
// path: đường dẫn REST (COMMIT_PATH hoặc FALL_EVENTS_PATH)
// body: chuỗi JSON
static bool fb_http_post(const char *path, const char *body)
{
    if (SIM_UART_LOCK() != pdTRUE) {
        ESP_LOGE(TAG, "Cannot get UART mutex");
        return false;
    }

    // 0. Giải phóng HTTP(S) session cũ nếu có (chống lỗi HTTP(S) busy)
    fb_at("AT+QHTTPSTOP", 2000, NULL);

    // 1. Cấu hình HTTP SSL (không xác minh cert — phù hợp cho đồ án)
    fb_at("AT+QHTTPCFG=\"contextid\",1",   3000, NULL);
    fb_at("AT+QHTTPCFG=\"sslctxid\",1",    3000, NULL);
    fb_at("AT+QHTTPCFG=\"contenttype\",4", 3000, NULL); // application/json
    fb_at("AT+QHTTPCFG=\"responseheader\",0", 3000, NULL);

    // SSL: tắt verify cert để tránh phức tạp (test mode)
    fb_at("AT+QSSLCFG=\"sslversion\",1,4",    3000, NULL); // TLS 1.2
    fb_at("AT+QSSLCFG=\"ciphersuite\",1,0xFFFF", 3000, NULL);
    fb_at("AT+QSSLCFG=\"ignorelocaltime\",1", 3000, NULL);
    fb_at("AT+QSSLCFG=\"seclevel\",1,0",      3000, NULL); // no cert check

    // 2. Set URL
    char url[512];
    snprintf(url, sizeof(url), "https://" FIREBASE_HOST "%s", path);

    char cmd[48];
    snprintf(cmd, sizeof(cmd), "AT+QHTTPURL=%d,%d", (int)strlen(url), 5);
    fb_flush();
    fb_sendline(cmd);
    fb_read(5000, "CONNECT");
    if (!strstr(s_rx_buf, "CONNECT")) {
        ESP_LOGE(TAG, "QHTTPURL: no CONNECT prompt");
        SIM_UART_UNLOCK();
        return false;
    }
    fb_send(url);
    fb_read(5000, "OK");

    // 3. POST body
    int bodylen = strlen(body);
    snprintf(cmd, sizeof(cmd), "AT+QHTTPPOST=%d,%d,%d",
             bodylen, HTTP_INPUT_TO, HTTP_RESP_TO);
    fb_flush();
    fb_sendline(cmd);
    fb_read(10000, "CONNECT");
    if (!strstr(s_rx_buf, "CONNECT")) {
        ESP_LOGE(TAG, "QHTTPPOST: no CONNECT prompt");
        SIM_UART_UNLOCK();
        return false;
    }

    ESP_LOGI(TAG, "Sending body (%d bytes)...", bodylen);
    fb_send(body);

    // 4. Chờ response
    fb_read(HTTP_RESP_TO * 1000, "+QHTTPPOST:");

    // HTTP 200 = thành công
    bool ok = strstr(s_rx_buf, ",200") != NULL;
    if (ok) {
        ESP_LOGI(TAG, "Firestore POST OK (HTTP 200)");
    } else {
        ESP_LOGE(TAG, "Firestore POST FAILED: %s", s_rx_buf);
    }

    SIM_UART_UNLOCK();
    return ok;
}

// ── Public API ────────────────────────────────────────────────────

void sim_firebase_init(void)
{
    ESP_LOGI(TAG, "Firebase component init (no extra setup needed)");
}

bool sim_firebase_update_status(const char *device_id,
                                 float lat,  float lon,
                                 float gps_accuracy,
                                 int   battery_pct,
                                 int   rssi,
                                 bool  fall_detected,
                                 float confidence,
                                 long  timestamp)
{
    // Dùng Firestore commit API (POST) để upsert document /devices/{device_id}
    // Không cần PATCH — commit với writes[].update là upsert.
    char doc_name[128];
    snprintf(doc_name, sizeof(doc_name),
             "projects/" FIREBASE_PROJECT_ID
             "/databases/(default)/documents/devices/%s",
             device_id ? device_id : "ESP32_FALL_001");

    // Build JSON body (~450 bytes)
    char body[768];
    snprintf(body, sizeof(body),
        "{\"writes\":[{\"update\":{"
            "\"name\":\"%s\","
            "\"fields\":{"
                "\"device_id\":{\"stringValue\":\"%s\"},"
                "\"lat\":{\"doubleValue\":%.6f},"
                "\"lng\":{\"doubleValue\":%.6f},"
                "\"gps_accuracy\":{\"doubleValue\":%.2f},"
                "\"battery_pct\":{\"integerValue\":\"%d\"},"
                "\"rssi\":{\"integerValue\":\"%d\"},"
                "\"fall_detected\":{\"booleanValue\":%s},"
                "\"confidence\":{\"doubleValue\":%.4f},"
                "\"timestamp\":{\"integerValue\":\"%ld\"}"
            "}"
        "}}]}",
        doc_name,
        device_id ? device_id : "ESP32_FALL_001",
        (double)lat, (double)lon,
        (double)gps_accuracy,
        battery_pct,
        rssi,
        fall_detected ? "true" : "false",
        (double)confidence,
        timestamp
    );

    ESP_LOGI(TAG, "Updating status to Firestore...");
    return fb_http_post(COMMIT_PATH, body);
}

bool sim_firebase_push_fall_event(const char *device_id,
                                   float lat,  float lon,
                                   float confidence,
                                   int   battery_pct,
                                   long  timestamp)
{
    // POST lên /fall_events → Firestore tự sinh document ID
    char body[512];
    snprintf(body, sizeof(body),
        "{\"fields\":{"
            "\"device_id\":{\"stringValue\":\"%s\"},"
            "\"lat\":{\"doubleValue\":%.6f},"
            "\"lng\":{\"doubleValue\":%.6f},"
            "\"confidence\":{\"doubleValue\":%.4f},"
            "\"battery_at_event\":{\"integerValue\":\"%d\"},"
            "\"timestamp\":{\"integerValue\":\"%ld\"}"
        "}}",
        device_id ? device_id : "ESP32_FALL_001",
        (double)lat, (double)lon,
        (double)confidence,
        battery_pct,
        timestamp
    );

    ESP_LOGI(TAG, "Pushing fall event to Firestore...");
    return fb_http_post(FALL_EVENTS_PATH, body);
}
