idf_component_register(
    SRCS "ai_model.cpp"
    INCLUDE_DIRS "include" "."
    REQUIRES
        espressif__esp-tflite-micro
)

target_compile_options(${COMPONENT_LIB} PRIVATE -std=c++17)