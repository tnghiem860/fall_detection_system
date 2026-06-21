#include <cstdint>
#include <cstring>
#include <cmath>

extern "C" {
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "mpu6050.h"
#include "sim_module.h"
#include "sim_mqtt.h"
#include "driver/gpio.h"
}

#include "ai_model.h"

static const char *TAG = "FALL_MAIN";

// ── Cấu hình ──────────────────────────────────────────────────────
#define PHONE_NUMBER            "0853344779"
#define DEVICE_ID               "ESP32_FALL_001"
#define WINDOW_SIZE             150
#define NUM_CHANNELS            3
#define POST_FALL_SAMPLES       25       // 75 mẫu sau impact
// FIX #1: Threshold tính trên ±16g scale — 1.6g² = 2.56 g²
#define IMPACT_AM2_THRESHOLD    2.56f                   // g² (AM >= 1.6g)
#define ALERT_COOLDOWN_MS       (30 * 1000)
#define SOS_BUTTON_GPIO         GPIO_NUM_25
#define BUZZER_GPIO             GPIO_NUM_13
// FIX #7: Chu kỳ báo cáo định kỳ (30 phút)
#define STATUS_REPORT_INTERVAL_MS  (30 * 60 * 1000)

// ── Ring buffer ────────────────────────────────────────────────────
static float ring_buf[WINDOW_SIZE][NUM_CHANNELS];
static int   ring_head  = 0;
static int   ring_count = 0;
// FIX (race): Mutex bảo vệ ring buffer khi nhiều task truy cập
static SemaphoreHandle_t s_ring_mutex = NULL;

// ── Post-fall state ────────────────────────────────────────────────
static bool impact_detected = false;
static int  post_fall_count = 0;

// FIX #5: s_buzzer_running + spinlock bao ve viet tu nhieu task
// (buzzer_countdown_task, sos_button_task, fall_detection_task cung doc/ghi)
static portMUX_TYPE  s_buzzer_mux     = portMUX_INITIALIZER_UNLOCKED;
static volatile bool s_buzzer_running = false;

// ── FIX #5: EventGroup dong bo ket qua buzzer countdown voi fall_task
// Bit 0 = countdown xong (khong huy), Bit 1 = nguoi dung huy
#define BUZZER_DONE_BIT    (1 << 0)
#define BUZZER_CANCEL_BIT  (1 << 1)
static EventGroupHandle_t s_buzzer_eg = NULL;

// ── FIX #5: SOS button ISR flag (volatile, doc tu sos_button_task)
static volatile bool s_sos_pending = false;

// ── FIX #7: Timer bao cao dinh ky ────────────────────────────────
static TimerHandle_t s_status_timer = NULL;
// Timer daemon task co stack nho, khong duoc goi UART blocking.
// fall_detection_task se doc flag va xu ly UART trong vong lap chinh.
static volatile bool s_status_pending = false;

// ── FIX: GPS Queue Pattern ────────────────────────────────────────
// on_gps_update chi enqueue ket qua, fall_detection_task xu ly MQTT.
// Tach biet GPS task va MQTT publishing de tranh tranh chap UART.
typedef struct {
    sim_gps_t gps;
    float     confidence;
} gps_alert_t;
static QueueHandle_t s_gps_result_queue = NULL;

// s_last_confidence: chuyen len global de on_gps_update tro cap gia tri
static float s_last_confidence = 0.0f;

// ── Build window từ ring buffer ───────────────────────────────────
static void build_window(float window[WINDOW_SIZE][NUM_CHANNELS])
{
    xSemaphoreTake(s_ring_mutex, portMAX_DELAY);
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (ring_head + i) % WINDOW_SIZE;
        window[i][0] = ring_buf[idx][0];
        window[i][1] = ring_buf[idx][1];
        window[i][2] = ring_buf[idx][2];
    }
    xSemaphoreGive(s_ring_mutex);
}

// ── FIX: GPS callback -> Queue Pattern ───────────────────────────
// Thay vi goi MQTT blocking truc tiep trong gps_background_task,
// on_gps_update chi gui ket qua vao Queue.
// fall_detection_task se nhan va xu ly MQTT trong vong lap cua minh,
// khong gay tranh chap UART giua hai task.
static void on_gps_update(sim_gps_t *gps)
{
    if (!gps || !s_gps_result_queue) return;

    gps_alert_t item;
    item.gps        = *gps;
    item.confidence = s_last_confidence;

    if (xQueueSend(s_gps_result_queue, &item, 0) != pdTRUE) {
        ESP_LOGW(TAG, "GPS result queue full — alert discarded");
    } else {
        ESP_LOGI(TAG, "GPS result queued (valid=%d, conf=%.2f)",
                 gps->valid, s_last_confidence);
    }
}

// ── GPIO init ─────────────────────────────────────────────────────
static void io_init(void)
{
    // Buzzer
    gpio_config_t buzzer_cfg = {};
    buzzer_cfg.pin_bit_mask  = 1ULL << BUZZER_GPIO;
    buzzer_cfg.mode          = GPIO_MODE_OUTPUT;
    buzzer_cfg.pull_up_en    = GPIO_PULLUP_DISABLE;
    buzzer_cfg.pull_down_en  = GPIO_PULLDOWN_DISABLE;
    buzzer_cfg.intr_type     = GPIO_INTR_DISABLE;
    gpio_config(&buzzer_cfg);
    gpio_set_level(BUZZER_GPIO, 0);

    // FIX #5: SOS button dùng External Interrupt + wakeup source
    // Không còn polling — ISR set flag, task xử lý
    gpio_config_t btn_cfg = {};
    btn_cfg.pin_bit_mask = 1ULL << SOS_BUTTON_GPIO;
    btn_cfg.mode         = GPIO_MODE_INPUT;
    btn_cfg.pull_up_en   = GPIO_PULLUP_ENABLE;
    btn_cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
    btn_cfg.intr_type    = GPIO_INTR_NEGEDGE;   // Ngắt cạnh xuống (nút nhấn)
    gpio_config(&btn_cfg);

    // Cho phép GPIO25 đánh thức ESP32 từ Light Sleep
    esp_sleep_enable_ext1_wakeup(1ULL << SOS_BUTTON_GPIO, ESP_EXT1_WAKEUP_ALL_LOW);
}

// ── FIX #5: SOS button ISR — chỉ set flag, không làm gì nặng trong ISR
static void IRAM_ATTR sos_isr_handler(void *arg)
{
    (void)arg;
    s_sos_pending = true;
}

static bool button_pressed(void)
{
    if (gpio_get_level(SOS_BUTTON_GPIO) == 0) {
        vTaskDelay(pdMS_TO_TICKS(50));  // debounce
        return gpio_get_level(SOS_BUTTON_GPIO) == 0;
    }
    return false;
}

// ── Buzzer countdown task ─────────────────────────────────────────
// FIX #5: Chạy trong task riêng để fall_detection_task không bị block 5s.
// Kết quả được báo qua EventGroup.
typedef struct {
    EventGroupHandle_t eg;
} buzzer_task_arg_t;

static void buzzer_countdown_task(void *arg)
{
    buzzer_task_arg_t *a = (buzzer_task_arg_t *)arg;
    // Lưu eg vào biến cục bộ TRƯỚC khi free(arg) để tránh use-after-free
    EventGroupHandle_t eg = a->eg;
    free(arg);

    gpio_set_level(BUZZER_GPIO, 1);

    for (int i = 0; i < 50; i++) {   // 50 x 100ms = 5 giay
        if (button_pressed()) {
            gpio_set_level(BUZZER_GPIO, 0);
            // Cho tha nut
            while (button_pressed()) vTaskDelay(pdMS_TO_TICKS(50));
            ESP_LOGW(TAG, "Canh bao bi huy boi nut nhan");
            // FIX: reset s_buzzer_running trong cancel path —
            // tránh stuck true mãi mãi khi countdown bị hủy giữa chừng
            taskENTER_CRITICAL(&s_buzzer_mux);
            s_buzzer_running = false;
            taskEXIT_CRITICAL(&s_buzzer_mux);
            xEventGroupSetBits(eg, BUZZER_CANCEL_BIT);
            vTaskDelete(NULL);
            return;
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    gpio_set_level(BUZZER_GPIO, 0);
    xEventGroupSetBits(eg, BUZZER_DONE_BIT);  // bao hieu fall_detection_task can gui alert
    taskENTER_CRITICAL(&s_buzzer_mux);         // FIX #5: spinlock bao ve s_buzzer_running
    s_buzzer_running = false;
    taskEXIT_CRITICAL(&s_buzzer_mux);
    vTaskDelete(NULL);
}

// FIX #5: Khoi dong buzzer countdown task + spinlock bao ve s_buzzer_running
static EventGroupHandle_t start_buzzer_countdown(void)
{
    taskENTER_CRITICAL(&s_buzzer_mux);
    s_buzzer_running = true;
    taskEXIT_CRITICAL(&s_buzzer_mux);

    buzzer_task_arg_t *arg = (buzzer_task_arg_t *)malloc(sizeof(buzzer_task_arg_t));
    if (!arg) {
        taskENTER_CRITICAL(&s_buzzer_mux);
        s_buzzer_running = false;
        taskEXIT_CRITICAL(&s_buzzer_mux);
        return NULL;
    }
    arg->eg = s_buzzer_eg;
    xEventGroupClearBits(s_buzzer_eg, BUZZER_DONE_BIT | BUZZER_CANCEL_BIT);
    xTaskCreate(buzzer_countdown_task, "buzzer_cd", 2048, arg, 6, NULL);
    return s_buzzer_eg;
}

// ── FIX #4: Gui canh bao SOS — non-blocking, dung GPS background task
// Fall_detection_task goi ham nay va lap tuc tra ve.
// Ket qua GPS duoc enqueue boi on_gps_update -> xu ly MQTT trong fall_detection_task.
static void trigger_alert_async(const char *reason, float confidence)
{
    ESP_LOGW(TAG, "Trigger alert: %s (conf=%.2f)", reason, confidence);

    vTaskDelay(pdMS_TO_TICKS(500));

    esp_err_t r = sim_gps_start_background(PHONE_NUMBER, on_gps_update);
    if (r == ESP_ERR_INVALID_STATE) {
        ESP_LOGW(TAG, "GPS task running, publish without location");
        sim_mqtt_ensure_connection(6);
        sim_mqtt_publish_fall_alert(DEVICE_ID, confidence, 0, 0, false);
    }
}

// ── FIX: Timer callback chi set flag + memory barrier ───────────
// Timer daemon task (stack ~2KB) KHONG duoc goi ham UART blocking.
// fall_detection_task se doc flag o dau vong lap while(1).
static void status_timer_cb(TimerHandle_t xTimer)
{
    (void)xTimer;
    s_status_pending = true;
    portMEMORY_BARRIER();  // FIX #4: dam bao flag visible tren ca hai core
}

// ── Init SIM, MQTT ────────────────────────────────────────────────
static void init_sim_and_mqtt(void)
{
    ESP_ERROR_CHECK(sim_init());
    ESP_ERROR_CHECK(sim_check_ready());

    sim_mqtt_init();

    while (!sim_mqtt_wait_network()) {
        ESP_LOGW(TAG, "MQTT network chưa sẵn sàng, thử lại 3s");
        vTaskDelay(pdMS_TO_TICKS(3000));
    }

    while (!sim_mqtt_start_pdp()) {
        ESP_LOGW(TAG, "PDP chưa có IP, thử lại 5s");
        vTaskDelay(pdMS_TO_TICKS(5000));
    }

    sim_mqtt_ensure_connection(6);
    sim_gps_init();
}

// ── Fall detection task (core 1) ──────────────────────────────────
// FIX #3: Chạy trên core 1. Light sleep (nếu dùng) chỉ ảnh hưởng core 1.
// sos_button_task và gps_background_task chạy trên core 0 — không bị ảnh hưởng.
static void fall_detection_task(void *arg)
{
    (void)arg;

    i2c_master_bus_handle_t bus_handle;
    ESP_ERROR_CHECK(i2c_master_init(&bus_handle));
    ESP_ERROR_CHECK(mpu6050_init(bus_handle));
    ESP_ERROR_CHECK(ai_model_init());

    init_sim_and_mqtt();
    sim_send_sms(PHONE_NUMBER, "HE THONG DA KHOI DONG!");
    sim_mqtt_publish_status(DEVICE_ID, false, 0.0f);

    vTaskDelay(pdMS_TO_TICKS(2000));
   // sim_sleep();

    ESP_LOGI(TAG, "=== Light Sleep mode: wakeup từ MPU6050 INT (GPIO%d) ===",
             MPU6050_INT_GPIO);

    TickType_t last_alert = xTaskGetTickCount() - pdMS_TO_TICKS(ALERT_COOLDOWN_MS);
    EventGroupHandle_t active_buzzer_eg = NULL;
    // s_last_confidence da la bien global — khoi tao o day
    s_last_confidence = 0.0f;

    // Buffer đọc FIFO — tối đa 170 mẫu (1024 byte / 6 byte)
    static mpu6050_data_t fifo_samples[170];

    while (1) {
        // ── Vào Light Sleep, thức khi MPU6050 báo FIFO gần đầy ──────
        // configure_light_sleep_wakeup đã set trong mpu6050_init()
        // Timer wakeup 4.5s làm backup phòng INT không kéo
        esp_sleep_enable_ext0_wakeup(MPU6050_INT_GPIO, 1);
    esp_sleep_enable_ext1_wakeup(1ULL << SOS_BUTTON_GPIO, ESP_EXT1_WAKEUP_ALL_LOW);
    esp_sleep_enable_timer_wakeup(4500 * 1000ULL);

    if (s_buzzer_running) {
        vTaskDelay(pdMS_TO_TICKS(100));
        continue;
    }
        //esp_light_sleep_start();

        // ── Thức dậy — lý do thức ────────────────────────────────────
        esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
        if (cause == ESP_SLEEP_WAKEUP_EXT0) {
            ESP_LOGD(TAG, "Wakeup: MPU FIFO INT");
        } else if (cause == ESP_SLEEP_WAKEUP_TIMER) {
            ESP_LOGD(TAG, "Wakeup: timer backup");
        }

        // ── FIX: Xu ly ket qua GPS tu Queue (non-blocking) ───────────────
        // gps_background_task gui ket qua vao queue; task nay xu ly MQTT.
        // Tach biet GPS callback va MQTT -> khong con tranh UART giua 2 task.
        gps_alert_t gps_alert;
        if (xQueueReceive(s_gps_result_queue, &gps_alert, 0) == pdTRUE) {
            ESP_LOGI(TAG, "Processing GPS alert from queue (valid=%d)", gps_alert.gps.valid);
            vTaskDelay(pdMS_TO_TICKS(500));  // cho modem on dinh
            sim_mqtt_ensure_connection(6);
            if (gps_alert.gps.valid) {
                sim_mqtt_publish_fall_alert(DEVICE_ID, gps_alert.confidence,
                                           (float)gps_alert.gps.lat,
                                           (float)gps_alert.gps.lon, true);
                sim_mqtt_publish_location(DEVICE_ID,
                                         (float)gps_alert.gps.lat,
                                         (float)gps_alert.gps.lon, "gps");
            } else {
                sim_mqtt_publish_fall_alert(DEVICE_ID, gps_alert.confidence,
                                           0, 0, false);
            }
        }

        // ── Xu ly status timer flag (tu 30 phut timer) ───────────────
        portMEMORY_BARRIER();  // FIX #4: dam bao doc gia tri moi nhat tu Timer task
        if (s_status_pending) {
            s_status_pending = false;
            if (sim_mqtt_ensure_connection(3)) {
                sim_mqtt_publish_status(DEVICE_ID, false, 0.0f);
                vTaskDelay(pdMS_TO_TICKS(1000));
            }
        }

        // ── Đọc hết FIFO ─────────────────────────────────────────────
        mpu6050_clear_interrupt();
        int n = 0;
        if (mpu6050_read(&fifo_samples[0]) == ESP_OK) {
        n = 1;
        }

        if (n <= 0) {
        ESP_LOGD(TAG, "Đọc mẫu thất bại — tiếp tục sleep");
        continue;
        }

        ESP_LOGD(TAG, "Đọc %d mẫu từ FIFO", n);

        // ── Xử lý từng mẫu tuần tự — giống vòng lặp 25Hz cũ ─────────
        for (int i = 0; i < n; i++) {
            mpu6050_data_t *imu = &fifo_samples[i];

            // Nạp vào ring buffer (đổi trục khớp dataset)
            xSemaphoreTake(s_ring_mutex, portMAX_DELAY);
            ring_buf[ring_head][0] = imu->ay_g;
            ring_buf[ring_head][1] = -imu->ax_g;
            ring_buf[ring_head][2] = imu->az_g;
            ring_head = (ring_head + 1) % WINDOW_SIZE;
            if (ring_count < WINDOW_SIZE) ring_count++;
            xSemaphoreGive(s_ring_mutex);

            float am2 = imu->ax_g * imu->ax_g +
                        imu->ay_g * imu->ay_g +
                        imu->az_g * imu->az_g;

            // Kiểm tra buzzer đang chạy không
            if (active_buzzer_eg != NULL) {
                EventBits_t bits = xEventGroupGetBits(active_buzzer_eg);
                if (bits & BUZZER_CANCEL_BIT) {
                    active_buzzer_eg = NULL;
                    if (sim_mqtt_ensure_connection(3)) {
                        sim_mqtt_publish_status(DEVICE_ID, false, 0.0f);
                    }
                } else if (bits & BUZZER_DONE_BIT) {
                    active_buzzer_eg = NULL;
                    trigger_alert_async("FALL DETECTED", s_last_confidence);
                }
            }

            // Impact detection
            if (!impact_detected &&
                am2 > IMPACT_AM2_THRESHOLD &&
                ring_count >= POST_FALL_SAMPLES) {
                impact_detected = true;
                post_fall_count = 0;
                ESP_LOGW(TAG, "Impact! AM2=%.3f", am2);
            }

            if (!impact_detected) continue;
            if (++post_fall_count < POST_FALL_SAMPLES) continue;

            if (ring_count < WINDOW_SIZE) {
                impact_detected = false;
                continue;
            }

            float window[WINDOW_SIZE][NUM_CHANNELS];
            build_window(window);
            impact_detected = false;

            ai_result_t res;
            if (ai_model_run(window, &res) != ESP_OK) continue;

            ESP_LOGI(TAG, "AI: fall=%d conf=%.3f", res.is_fall, res.confidence);
            if (!res.is_fall) continue;
            if (res.confidence < 0.70f) {
            ESP_LOGW(TAG, "Fall detected nhưng confidence %.1f%% < 70%% — bỏ qua",
             res.confidence * 100.0f);
            continue;
            }           

            TickType_t now = xTaskGetTickCount();
            if ((now - last_alert) < pdMS_TO_TICKS(ALERT_COOLDOWN_MS)) {
                ESP_LOGW(TAG, "Cooldown chưa hết");
                continue;
            }

            last_alert = now;
            s_last_confidence = res.confidence;
            active_buzzer_eg  = start_buzzer_countdown();

            // Có fall — thoát vòng for, xử lý alert trước
            // Các mẫu còn lại trong FIFO batch này bỏ qua
            break;
        }
    }
}
// ── SOS button task (core 0) ──────────────────────────────────────
// FIX #5: Dùng ISR flag thay vì polling.
// Task chờ flag từ ISR, xử lý debounce, rồi trigger alert.
static void sos_button_task(void *arg)
{
    (void)arg;

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(50));

        if (!s_sos_pending) continue;
        s_sos_pending = false;

        // FIX: Nếu buzzer countdown đang chạy, lần bấm này dùng để HỦY còi
        // chứ không phải kích SOS — bỏ qua để tránh kích hoạt SOS sai
        bool buzzer_was_running;
        taskENTER_CRITICAL(&s_buzzer_mux);
        buzzer_was_running = s_buzzer_running;
        taskEXIT_CRITICAL(&s_buzzer_mux);
        if (buzzer_was_running) {
            ESP_LOGI(TAG, "Nút nhấn đã dùng để hủy còi té ngã — bỏ qua SOS");
            continue;
        }

        vTaskDelay(pdMS_TO_TICKS(50));
        if (!button_pressed()) continue;

        ESP_LOGW(TAG, "SOS thủ công được nhấn");

        while (button_pressed()) vTaskDelay(pdMS_TO_TICKS(50));
        vTaskDelay(pdMS_TO_TICKS(300));

        s_buzzer_running = true;
        gpio_set_level(BUZZER_GPIO, 1);

        // FIX: Dùng timestamp thay vì vòng for đếm — chính xác hơn
        // không bị ảnh hưởng bởi preemption hay delay bên ngoài
        bool cancelled = false;
        TickType_t sos_start = xTaskGetTickCount();

        while (xTaskGetTickCount() - sos_start < pdMS_TO_TICKS(5000)) {
            if (button_pressed()) {
                while (button_pressed()) vTaskDelay(pdMS_TO_TICKS(50));
                ESP_LOGW(TAG, "SOS hủy bởi người dùng");
                cancelled = true;
                break;
            }
            vTaskDelay(pdMS_TO_TICKS(20));  // poll nhanh hơn — 20ms thay vì 100ms
        }

        gpio_set_level(BUZZER_GPIO, 0);
        taskENTER_CRITICAL(&s_buzzer_mux);  // FIX #5
        s_buzzer_running = false;
        taskEXIT_CRITICAL(&s_buzzer_mux);

        if (cancelled) {
            // Sleep đang disable — modem luôn active, không cần wakeup
            vTaskDelay(pdMS_TO_TICKS(200));
            if (sim_mqtt_ensure_connection(3)) {
                sim_mqtt_publish_status(DEVICE_ID, false, 0.0f);
            }
        } else {
            trigger_alert_async("MANUAL SOS", 1.0f);
        }
    }
}

// ── App main ──────────────────────────────────────────────────────
extern "C" void app_main(void)
{
    // Khởi tạo GPIO sớm nhất (buzzer + button config)
    io_init();

    // FIX LỖI 2: Đăng ký ISR service và handler tại đây — chỉ gọi một lần
    // toàn hệ thống. Không gọi lại trong bất kỳ task nào.
    ESP_ERROR_CHECK(gpio_install_isr_service(0));
    ESP_ERROR_CHECK(gpio_isr_handler_add(SOS_BUTTON_GPIO, sos_isr_handler, NULL));

    // Khởi tạo FreeRTOS primitives
    s_ring_mutex = xSemaphoreCreateMutex();
    configASSERT(s_ring_mutex);

    s_buzzer_eg = xEventGroupCreate();
    configASSERT(s_buzzer_eg);

    // FIX: Khoi tao GPS result queue (capacity=2 de khong mat ket qua neu task busy)
    s_gps_result_queue = xQueueCreate(2, sizeof(gps_alert_t));
    configASSERT(s_gps_result_queue);

    // FIX #7: Timer báo cáo định kỳ (30 phút, auto-reload)
    s_status_timer = xTimerCreate("status_rpt",
                                  pdMS_TO_TICKS(STATUS_REPORT_INTERVAL_MS),
                                  pdTRUE,   // auto-reload
                                  NULL,
                                  status_timer_cb);
    configASSERT(s_status_timer);
    xTimerStart(s_status_timer, 0);

    // FIX #3: fall_detection_task pin vào core 1
    //         sos_button_task và gps_background_task (tạo trong sim_module.c)
    //         chạy trên core 0 — UART và GPS không bị ảnh hưởng khi core 1 sleep
    xTaskCreatePinnedToCore(fall_detection_task, "fall_det",
                            16384, NULL, 5, NULL, 1);

    // FIX #5: sos_button_task trên core 0 — nhận ISR flag, xử lý SOS
    xTaskCreatePinnedToCore(sos_button_task, "sos_btn",
                            4096, NULL, 7, NULL, 0);
}