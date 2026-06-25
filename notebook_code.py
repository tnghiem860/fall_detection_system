
!pip install -q tensorflow-model-optimization tf-keras

import os
# TFMOT 0.8.x ổn định hơn khi dùng tf.keras legacy, đặc biệt trên Kaggle TF 2.18/2.19
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score,
                              recall_score, f1_score)
from sklearn.utils import resample
import tensorflow as tf
import tensorflow_model_optimization as tfmot
print('TF version :', tf.__version__)
print('TFMOT version:', tfmot.__version__)

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
DATASET_ROOT  = '/kaggle/input/datasets/nvnikhil0001/sis-fall-original-dataset/SisFall_dataset'
OUTPUT_DIR    = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tự tìm lại thư mục SisFall_dataset nếu Kaggle mount dataset ở đường dẫn khác
if not os.path.isdir(DATASET_ROOT):
    candidates = glob.glob('/kaggle/input/**/SisFall_dataset', recursive=True)
    if candidates:
        DATASET_ROOT = candidates[0]
        print('Đã tự tìm thấy DATASET_ROOT:', DATASET_ROOT)
    else:
        print('Không tìm thấy SisFall_dataset trong /kaggle/input. Hãy kiểm tra lại dataset đã Add vào notebook chưa.')

# Pre-processing (theo paper Section 2.3)
ORIG_FS       = 200          # Hz - tần số gốc SisFall
TARGET_FS     = 25           # Hz - down-sample về 25 Hz
DOWNSAMPLE_N  = ORIG_FS // TARGET_FS   # = 8
AM_THRESHOLD  = 1.6          # g   - ngưỡng impact point
WINDOW_S      = 6            # giây
WINDOW_LEN    = TARGET_FS * WINDOW_S   # = 150 samples

# Nhãn file: tiền tố 'D' = ADL, 'F' = Fall (theo SisFall naming convention)
FALL_PREFIX   = 'F'
ADL_PREFIX    = 'D'

# Training
EPOCHS        = 85
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
DROPOUT_RATE  = 0.4

# QAT
QAT_EPOCHS    = 5           # fine-tune thêm sau khi áp QAT
QAT_LR        = 1e-5         # LR nhỏ hơn cho giai đoạn QAT

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
print('Config OK')
print('DATASET_ROOT =', DATASET_ROOT)

# ─── Ma trận chuyển trục theo vị trí đặt cảm biến ────────────────────────────
#
# SisFall đặt sensor ở bụng trước:
#   X sang trái, Y xuống dưới, Z ra trước (hướng dương).
#
# Khi đặt cảm biến ở vị trí khác, ta xoay sensor quanh trục Y (dọc cơ thể):
#   front (0°)  : giữ nguyên      → [X,  Y,  Z]
#   right (−90°): xoay sang phải  → [−Z, Y,  X]
#   left  (+90°): xoay sang trái  → [+Z, Y, −X]
#   back  (180°): xoay ra sau     → [−X, Y, −Z]
#
# Ma trận 3×3 mỗi hàng = [hệ số trục X gốc, trục Y gốc, trục Z gốc]
# để tạo ra trục mới tương ứng.

POSITION_TRANSFORMS = {
    'front': np.array([[ 1,  0,  0],   # X_new =  X_old
                       [ 0,  1,  0],   # Y_new =  Y_old
                       [ 0,  0,  1]], dtype=np.float32),  # Z_new =  Z_old

    'right': np.array([[ 0,  0, -1],   # X_new = -Z_old
                       [ 0,  1,  0],   # Y_new =  Y_old
                       [ 1,  0,  0]], dtype=np.float32),  # Z_new =  X_old

    'left' : np.array([[ 0,  0,  1],   # X_new = +Z_old
                       [ 0,  1,  0],   # Y_new =  Y_old
                       [-1,  0,  0]], dtype=np.float32),  # Z_new = -X_old

    'back' : np.array([[-1,  0,  0],   # X_new = -X_old
                       [ 0,  1,  0],   # Y_new =  Y_old
                       [ 0,  0, -1]], dtype=np.float32),  # Z_new = -Z_old
}

# Danh sách các vị trí sẽ dùng để train (có thể bỏ bớt nếu muốn)
POSITIONS = ['front', 'right', 'left', 'back']

print('Đã định nghĩa POSITION_TRANSFORMS cho:', POSITIONS)


def apply_position_transform(accel_raw, position):
    """
    Áp dụng ma trận xoay để chuyển tín hiệu gốc SisFall (bụng trước)
    sang hệ trục của vị trí đặt cảm biến mong muốn.

    Parameters
    ----------
    accel_raw : np.ndarray, shape (N, 3)  — tín hiệu gốc đơn vị g
    position  : str — 'front' | 'right' | 'left' | 'back'

    Returns
    -------
    np.ndarray, shape (N, 3) — tín hiệu đã chuyển trục
    """
    if position not in POSITION_TRANSFORMS:
        raise ValueError(f'position phải là một trong {list(POSITION_TRANSFORMS.keys())}, nhận được: {position!r}')
    M = POSITION_TRANSFORMS[position]   # (3, 3)
    return accel_raw @ M.T              # (N, 3) @ (3, 3) → (N, 3)


def parse_sisfall_file(filepath, position='front'):
    """
    SisFall format: mỗi dòng = 9 giá trị int16, phân cách bởi dấu phẩy.
    Cột 0-2  : ADXL345 accelerometer 1  (raw counts, ±16 g, 13-bit)
    Cột 3-5  : ITG3200 gyroscope (KHÔNG dùng)
    Cột 6-8  : MMA8451Q accelerometer 2 (KHÔNG dùng)

    Chuyển đổi: 1 count ADXL345 = (16×2) / 2^13 ≈ 0.00390625 g
    Sau đó áp ma trận xoay theo position.
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if not line:
                continue
            parts = re.split(r'[,;\s]+', line)
            if len(parts) >= 3:
                try:
                    vals = [float(p) for p in parts[:9]]
                    rows.append(vals)
                except ValueError:
                    continue
    if not rows:
        return None
    data = np.array(rows)                            # (N, 9)
    accel_raw = data[:, 0:3] * (32.0 / 8192.0)     # (N, 3) đơn vị g, hệ trục SisFall gốc
    accel = apply_position_transform(accel_raw, position)
    return accel  # (N, 3) đã chuyển sang hệ trục của vị trí yêu cầu


def acceleration_magnitude(accel):
    """AM = sqrt(ax^2 + ay^2 + az^2) — Equation (2) trong paper"""
    return np.sqrt(np.sum(accel**2, axis=1))


def downsample(accel, factor):
    """Down-sample bằng cách lấy mẫu theo bước (bỏ qua LPF cho đơn giản)"""    
    return accel[::factor]


def extract_window(accel_ds):
    """
    Tìm impact point: vị trí AM vượt ngưỡng AM_THRESHOLD.
    Cắt cửa sổ 6 s (150 samples @ 25 Hz) căn giữa impact point.
    AM được tính trên toàn 3 trục — không phụ thuộc vị trí đặt cảm biến.
    """
    am = acceleration_magnitude(accel_ds[:, 0:3])
    candidates = np.where(am > AM_THRESHOLD)[0]
    if len(candidates) == 0:
        peak = int(np.argmax(am))
    else:
        peak = int(candidates[np.argmax(am[candidates])])

    half  = WINDOW_LEN // 2
    start = peak - half
    end   = start + WINDOW_LEN

    n = len(accel_ds)
    if start < 0:
        start = 0
        end   = WINDOW_LEN
    if end > n:
        end   = n
        start = max(0, n - WINDOW_LEN)

    window = accel_ds[start:end]
    if len(window) < WINDOW_LEN:
        pad = np.zeros((WINDOW_LEN - len(window), 3))
        window = np.vstack([window, pad])
    return window  # (150, 3)


print('Functions parse_sisfall_file / extract_window defined.')

def get_subject_id(filepath):
    """
    Lấy mã người từ đường dẫn hoặc tên file.
    Ví dụ SisFall: .../SA01/D04_SA01_R01.txt  → SA01
                   .../SE03/F01_SE03_R01.txt  → SE03
    """
    fname  = os.path.basename(filepath)
    parent = os.path.basename(os.path.dirname(filepath))

    if re.match(r'^(SA|SE)\d+$', parent):
        return parent

    m = re.search(r'_(SA|SE)\d+_', fname)
    if m:
        return m.group(0).strip('_')

    return parent


def load_sisfall_multi(root, positions=POSITIONS):
    """
    Quét toàn bộ thư mục SisFall và sinh dữ liệu cho NHIỀU vị trí đặt cảm biến.

    Với mỗi file .txt hợp lệ, hàm tạo ra len(positions) mẫu — mỗi mẫu
    tương ứng với một phép xoay trục (vị trí đặt) khác nhau. Điều này
    giúp model học được đặc trưng ngã không phụ thuộc vào vị trí gắn sensor.

    Trả về
    -------
    X              : np.ndarray (N, 150, 3)
    y              : np.ndarray (N,)  — nhãn 0=ADL / 1=Fall
    groups         : np.ndarray (N,)  — subject_id, dùng để GroupSplit
    position_labels: np.ndarray (N,)  — tên vị trí ('front','right','left','back')
    """
    X, y, groups, position_labels = [], [], [], []

    txt_files = glob.glob(os.path.join(root, '**', '*.txt'), recursive=True)
    print(f'Tìm thấy {len(txt_files)} file .txt')

    if len(txt_files) == 0:
        raise FileNotFoundError(
            f'Không tìm thấy file .txt trong DATASET_ROOT={root}. '
            'Hãy kiểm tra đường dẫn dataset hoặc Add dataset SisFall vào Kaggle notebook.'
        )

    skipped = 0
    for fp in txt_files:
        fname = os.path.basename(fp)

        if fname.startswith(FALL_PREFIX):
            label = 1
        elif fname.startswith(ADL_PREFIX):
            label = 0
        else:
            continue

        subject_id = get_subject_id(fp)

        # Đọc raw accel một lần, sau đó áp từng transform
        rows = []
        with open(fp, 'r') as fh:
            for line in fh:
                line = line.strip().rstrip(',')
                if not line:
                    continue
                parts = re.split(r'[,;\s]+', line)
                if len(parts) >= 3:
                    try:
                        vals = [float(p) for p in parts[:9]]
                        rows.append(vals)
                    except ValueError:
                        continue

        if not rows:
            skipped += 1
            continue

        data      = np.array(rows)
        accel_raw = data[:, 0:3] * (32.0 / 8192.0)   # (N, 3) đơn vị g

        if len(accel_raw) < DOWNSAMPLE_N * 20:
            skipped += 1
            continue

        for pos in positions:
            accel_pos = apply_position_transform(accel_raw, pos)
            accel_ds  = downsample(accel_pos, DOWNSAMPLE_N)
            window    = extract_window(accel_ds)           # (150, 3)

            X.append(window)
            y.append(label)
            groups.append(subject_id)
            position_labels.append(pos)

    X               = np.array(X,               dtype=np.float32)
    y               = np.array(y,               dtype=np.int32)
    groups          = np.array(groups)
    position_labels = np.array(position_labels)

    if len(X) == 0:
        raise ValueError('Không load được mẫu hợp lệ nào từ SisFall.')

    if len(np.unique(y)) < 2:
        raise ValueError('Dataset sau khi load chỉ có 1 class. Cần đủ cả ADL và Fall.')

    print(f'Load xong ({len(positions)} vị trí × file): {len(X)} mẫu tổng, bỏ qua {skipped} file')
    for pos in positions:
        mask = position_labels == pos
        print(f'  [{pos:5s}] tổng={mask.sum()}  ADL={np.sum(y[mask]==0)}  Fall={np.sum(y[mask]==1)}')
    print(f'  Số subject: {len(np.unique(groups))}')

    return X, y, groups, position_labels


# ── Chạy load ────────────────────────────────────────────────────────────────
X_raw, y_raw, groups_raw, pos_raw = load_sisfall_multi(DATASET_ROOT, positions=POSITIONS)

X_cnn = X_raw[..., np.newaxis]   # (N, 150, 3, 1)

print(f'\nX_cnn shape      : {X_cnn.shape}')
print(f'groups_raw shape : {groups_raw.shape}')
print(f'pos_raw shape    : {pos_raw.shape}')

# ─── 3.1 Split DỮ LIỆU THEO NGƯỜI trước khi normalize ───────────────────────
#
# Lưu ý quan trọng: mỗi subject giờ xuất hiện trong CẢ 4 vị trí.
# GroupShuffleSplit vẫn đảm bảo không rò rỉ subject — khi 1 subject vào
# tập test thì TẤT CẢ 4 vị trí của subject đó đều vào test.
from sklearn.model_selection import GroupShuffleSplit

groups_raw = np.array(groups_raw)


def has_both_classes(y_part):
    return len(np.unique(y_part)) == 2


def group_split_with_class_check(X, y, groups, test_size, seed, max_tries=200):
    """
    Chia theo group/subject — đảm bảo không trùng subject và cả 2 tập có đủ 2 class.
    Trả về (idx_a, idx_b).
    """
    for s in range(seed, seed + max_tries):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
        idx_a, idx_b = next(splitter.split(X, y, groups=groups))
        if has_both_classes(y[idx_a]) and has_both_classes(y[idx_b]):
            return idx_a, idx_b
    raise ValueError(
        'Không tìm được cách chia group sao cho cả hai tập đều có đủ ADL và Fall. '
        'Hãy giảm test_size/val_size hoặc kiểm tra phân bố dữ liệu theo subject.'
    )


# Bước 1: tách Test 30% theo subject
idx_temp, idx_test = group_split_with_class_check(
    X_cnn, y_raw, groups_raw, test_size=0.30, seed=SEED
)

X_temp      = X_cnn[idx_temp];      y_temp      = y_raw[idx_temp]
groups_temp = groups_raw[idx_temp]; pos_temp    = pos_raw[idx_temp]

X_test      = X_cnn[idx_test];      y_test      = y_raw[idx_test]
groups_test = groups_raw[idx_test]; pos_test    = pos_raw[idx_test]


# Bước 2: từ 70% còn lại tách Val ≈ 10% toàn bộ dataset
val_ratio_inside_temp = 0.10 / 0.70

idx_train_local, idx_val_local = group_split_with_class_check(
    X_temp, y_temp, groups_temp, test_size=val_ratio_inside_temp, seed=SEED + 1000
)

X_train_unbal  = X_temp[idx_train_local]
y_train_unbal  = y_temp[idx_train_local]
groups_train   = groups_temp[idx_train_local]
pos_train_unbal = pos_temp[idx_train_local]

X_val          = X_temp[idx_val_local]
y_val          = y_temp[idx_val_local]
groups_val     = groups_temp[idx_val_local]
pos_val        = pos_temp[idx_val_local]


# ─── Kiểm tra rò rỉ subject ────────────────────────────────────────────────
train_subjects = set(groups_train)
val_subjects   = set(groups_val)
test_subjects  = set(groups_test)

print('Kiểm tra trùng subject:')
print('  Train ∩ Val :', train_subjects & val_subjects)
print('  Train ∩ Test:', train_subjects & test_subjects)
print('  Val ∩ Test  :', val_subjects & test_subjects)

if train_subjects & val_subjects:
    raise ValueError('Lỗi: Train và Val bị trùng subject.')
if train_subjects & test_subjects:
    raise ValueError('Lỗi: Train và Test bị trùng subject.')
if val_subjects & test_subjects:
    raise ValueError('Lỗi: Val và Test bị trùng subject.')

print()
print('Trước augmentation (tất cả vị trí gộp):')
print(f'  Train : {X_train_unbal.shape[0]:5d} mẫu | Subject: {len(train_subjects)} | ADL: {np.sum(y_train_unbal==0)} | Fall: {np.sum(y_train_unbal==1)}')
print(f'  Val   : {X_val.shape[0]:5d} mẫu | Subject: {len(val_subjects)} | ADL: {np.sum(y_val==0)} | Fall: {np.sum(y_val==1)}')
print(f'  Test  : {X_test.shape[0]:5d} mẫu | Subject: {len(test_subjects)} | ADL: {np.sum(y_test==0)} | Fall: {np.sum(y_test==1)}')

print()
print('Breakdown theo vị trí (train unbalanced):')
for pos in POSITIONS:
    mask = pos_train_unbal == pos
    print(f'  [{pos:5s}] ADL={np.sum(y_train_unbal[mask]==0):4d}  Fall={np.sum(y_train_unbal[mask]==1):4d}')


# ─── 3.2 Normalize Accelerometer (chỉ tính trên tập train) ────────────────
x_min_global = float(X_train_unbal.min())
x_max_global = float(X_train_unbal.max())

if np.isclose(x_max_global, x_min_global):
    raise ValueError('Giá trị max bằng min ở Accel, không thể normalize.')

def normalize_accel(X, x_min, x_max):
    """Chuẩn hóa về [-1, 1] theo min/max của tập train."""   
    return 2.0 * (X - x_min) / (x_max - x_min) - 1.0

X_train_unbal = normalize_accel(X_train_unbal, x_min_global, x_max_global).astype(np.float32)
X_val         = normalize_accel(X_val,         x_min_global, x_max_global).astype(np.float32)
X_test        = normalize_accel(X_test,        x_min_global, x_max_global).astype(np.float32)

print()
print(f'Normalize range (từ train): [{x_min_global:.4f} g, {x_max_global:.4f} g]')
print('Lưu lại 2 giá trị này để dùng trên ESP32.')

# ─── 3.3 & 3.4  Cân bằng dữ liệu bằng Data Augmentation (Jittering + Scaling) ──
#
# Nhờ nhân 4 vị trí, số mẫu Fall tăng lên đáng kể so với bản gốc,
# tuy nhiên ADL vẫn chiếm đa số. Ta augment Fall để cân bằng hoàn toàn.

idx_fall_train = np.where(y_train_unbal == 1)[0]
idx_adl_train  = np.where(y_train_unbal == 0)[0]

majority_n = len(idx_adl_train)
minority_n = len(idx_fall_train)

print(f'Trước augment — ADL: {majority_n}  Fall: {minority_n}  (tỉ lệ {majority_n/minority_n:.2f}x)')


def augment_timeseries(X, num_required):
    """
    Sinh thêm dữ liệu Fall để cân bằng với ADL, dùng 2 kỹ thuật:
      - Jittering : cộng nhiễu Gaussian (loc=0.0, scale=0.02) vào toàn bộ tín hiệu,
                    giả lập nhiễu điện từ môi trường hoặc sai số phần cứng cảm biến.
      - Scaling   : nhân biên độ 3 trục X/Y/Z với hệ số ngẫu nhiên trong [0.9, 1.1],
                    giả lập lực va đập khi ngã mạnh/nhẹ hơn thực tế phòng lab.
    Hai phép biến đổi áp dụng tuần tự trên mỗi mẫu sinh ra.
    """
    augmented = []
    n_samples = len(X)
    for _ in range(num_required):
        sample = X[np.random.randint(0, n_samples)].copy()

        # 1. Jittering
        noise  = np.random.normal(loc=0.0, scale=0.02, size=sample.shape)
        sample = sample + noise

        # 2. Scaling
        scale_factor = np.random.uniform(0.9, 1.1)
        sample = sample * scale_factor

        augmented.append(sample)
    return np.array(augmented, dtype=np.float32)


if minority_n < majority_n:
    num_to_add = majority_n - minority_n
    print(f'Đang augment thêm {num_to_add} mẫu Fall...')

    X_fall_base = X_train_unbal[idx_fall_train]
    pos_fall_base = pos_train_unbal[idx_fall_train]

    X_fall_aug = augment_timeseries(X_fall_base, num_to_add)

    # Gán position_label cho mẫu augment: lặp vòng tròn từ mẫu gốc
    pos_fall_aug = np.array([
        pos_fall_base[i % len(pos_fall_base)] for i in range(num_to_add)
    ])

    X_train = np.vstack([X_train_unbal[idx_adl_train], X_fall_base, X_fall_aug])
    y_train = np.hstack([
        np.zeros(majority_n, dtype=np.int32),
        np.ones(minority_n + num_to_add, dtype=np.int32)
    ])
    pos_train = np.concatenate([
        pos_train_unbal[idx_adl_train],
        pos_fall_base,
        pos_fall_aug
    ])

    from sklearn.utils import shuffle
    X_train, y_train, pos_train = shuffle(X_train, y_train, pos_train, random_state=SEED)
else:
    X_train   = X_train_unbal
    y_train   = y_train_unbal
    pos_train = pos_train_unbal

print(f'\nSau augmentation tập Train:')
print(f'  ADL (0): {np.sum(y_train==0)}   Fall (1): {np.sum(y_train==1)}')
print(f'  Tổng   : {len(y_train)} mẫu')
print()
print('Breakdown theo vị trí (train sau augment):')
for pos in POSITIONS:
    mask = pos_train == pos
    print(f'  [{pos:5s}] ADL={np.sum(y_train[mask]==0):5d}  Fall={np.sum(y_train[mask]==1):5d}')
print()
print('Test set (không augment):')
print(f'  ADL (0): {np.sum(y_test==0)}   Fall (1): {np.sum(y_test==1)}')

def build_fdcnn_lightweight(input_shape=(WINDOW_LEN, 3, 1),
                            dropout_rate=DROPOUT_RATE,
                            num_classes=2):
    """
    FD-CNN phiên bản "Lai" siêu nhẹ cho ESP32.
    Dùng GlobalAveragePooling2D thay cho Flatten để ép xung dung lượng.
    """
    inp = tf.keras.Input(shape=input_shape, name='input')

    # ── Block 1: Conv2D(16) ─────────────────────────────────────────────────
    x = tf.keras.layers.Conv2D(16, kernel_size=(1, 3), strides=(1, 1),
                               padding='same', use_bias=False, name='conv1')(inp)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='pool1')(x)

    # ── Block 2: Conv2D(32) ─────────────────────────────────────────────────
    x = tf.keras.layers.Conv2D(32, kernel_size=(1, 3), strides=(1, 1),
                               padding='same', use_bias=False, name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='pool2')(x)

    # ── Block 3: Conv2D(64) ─────────────────────────────────────────────────
    x = tf.keras.layers.Conv2D(64, kernel_size=(1, 3), strides=(1, 1),
                               padding='same', use_bias=False, name='conv3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.ReLU(name='relu3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='pool3')(x)

    # =========================================================================
    # ── PHẦN THAY ĐỔI CHÍNH Ở ĐÂY ──
    # =========================================================================
    
    # 1. Dùng GlobalAverage thay vì Flatten
    # Thay vì để lại 3456 giá trị (18x3x64), lớp này ép lại chỉ còn đúng 64 giá trị.
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # 2. Dropout để chống overfit
    x = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')(x)

    # 3. Thu nhỏ lớp Dense (giống code cũ của bạn)
    # Bỏ lớp 512 và 32, chỉ dùng 1 lớp Dense(16) cực nhẹ
    x = tf.keras.layers.Dense(16, use_bias=True, name='dense_light')(x)
    x = tf.keras.layers.ReLU(name='relu_light')(x)

    # ── Output ───────────────────────────────────────────────────────────────
    out = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name='FD_CNN_Light')
    return model
# SAI (đang gọi lại hàm cũ): 
# base_model = build_fdcnn()

# ĐÚNG (gọi hàm siêu nhẹ):
base_model = build_fdcnn_lightweight() 

base_model.summary() # Dòng này in ra bảng summary để bạn check lại
base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

best_base_path = os.path.join(OUTPUT_DIR, 'best_base_model.weights.h5')
callbacks_base = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_base_path,
        monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=0)
]

print('=== GIAI ĐOẠN 1: Train float32 base model ===')
history_base = base_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_base,
    verbose=1
)

# ── Plot training curves ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_base.history['accuracy'],     label='Train Acc')
axes[0].plot(history_base.history['val_accuracy'], label='Val Acc')
axes[0].set_title('Float32 — Accuracy'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(history_base.history['loss'],     label='Train Loss')
axes[1].plot(history_base.history['val_loss'], label='Val Loss')
axes[1].set_title('Float32 — Loss'); axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_float32.png'), dpi=150)
plt.show()
# ── Nạp best float32 weights ──────────────────────────────────────────────
best_base_path = os.path.join(OUTPUT_DIR, 'best_base_model.weights.h5')
if os.path.exists(best_base_path):
    base_model.load_weights(best_base_path)
    print('Đã load best float32 weights.')
else:
    print('Dùng weights cuối cùng của base model.')

# ── Áp QAT lên toàn bộ model ──────────────────────────────────────────────
quantize_model = tfmot.quantization.keras.quantize_model
qat_model = quantize_model(base_model)

qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=QAT_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
qat_model.summary()

best_qat_path = os.path.join(OUTPUT_DIR, 'best_qat_model.weights.h5')
callbacks_qat = [
    # Giảm patience xuống 2 vì QAT epoch ít
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
    # Theo dõi val_loss thay vì val_accuracy cho QAT để tránh overfit sát nút
    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_qat_path,
        monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
]

print('=== GIAI ĐOẠN 2: Quantization-Aware Training (fake int8) ===')
history_qat = qat_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=QAT_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_qat,
    verbose=1
)
# Plot QAT curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_qat.history['accuracy'],     label='Train Acc')
axes[0].plot(history_qat.history['val_accuracy'], label='Val Acc')
axes[0].set_title('QAT — Accuracy'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(history_qat.history['loss'],     label='Train Loss')
axes[1].plot(history_qat.history['val_loss'], label='Val Loss')
axes[1].set_title('QAT — Loss'); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_qat.png'), dpi=150)
plt.show()
# ── Nạp best QAT weights ─────────────────────────────────────────────────
best_qat_path = os.path.join(OUTPUT_DIR, 'best_qat_model.weights.h5')
if os.path.exists(best_qat_path):
    qat_model.load_weights(best_qat_path)
    print('Đã load best QAT weights.')
else:
    print('Không tìm thấy best_qat_model.weights.h5 → dùng qat_model hiện tại.')

# ── Representative dataset cho full-integer calibration ──────────────────
def representative_data_gen():
    """Dùng subset của tập train để hiệu chỉnh scale factor int8."""
    indices = np.random.choice(len(X_train), size=min(500, len(X_train)), replace=False)
    for i in indices:
        sample = X_train[i:i+1].astype(np.float32)  # (1, 150, 3, 1)
        yield [sample]

# ── TFLite Converter ────────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

print('Đang convert sang TFLite int8...')
tflite_int8_model = converter.convert()

# ── Lưu file .tflite ────────────────────────────────────────────────────
tflite_path = os.path.join(OUTPUT_DIR, 'fd_cnn_multipos_int8.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_int8_model)

size_kb = os.path.getsize(tflite_path) / 1024
print(f'\nFile TFLite int8 đa vị trí đã lưu: {tflite_path}')
print(f'Kích thước: {size_kb:.1f} KB')

# ─── 8.1  Đánh giá QAT model (float-compatible) trên tập test ────────────
y_pred_prob = qat_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

# Fix: dùng labels=[0,1] để tránh lỗi khi model predict 1 class duy nhất
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
spec_tn, spec_fp, spec_fn, spec_tp = cm.ravel()
spec = spec_tn / (spec_tn + spec_fp) if (spec_tn + spec_fp) > 0 else 0.0

print('=' * 50)
print('     KẾT QUẢ QAT MODEL (test set)')
print('=' * 50)
print(f'  Accuracy    : {acc*100:.2f}%')
print(f'  Precision   : {prec*100:.2f}%')
print(f'  Sensitivity : {rec*100:.2f}%')
print(f'  Specificity : {spec*100:.2f}%')
print(f'  F1-Score    : {f1:.4f}')
print('=' * 50)
print()
print(classification_report(y_test, y_pred, target_names=['ADL', 'Fall'], zero_division=0))
# ─── 8.2  Confusion Matrix ────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['ADL','Fall']); ax.set_yticklabels(['ADL','Fall'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix — QAT Model')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_qat.png'), dpi=150)
plt.show()

# ─── 8.3  Đánh giá TFLite int8 model bằng Interpreter ────────────────────
print('=== Kiểm tra TFLite int8 model ===')

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('Input  dtype:', input_details[0]['dtype'])
print('Output dtype:', output_details[0]['dtype'])
print('Input  shape:', input_details[0]['shape'])

# Scale và zero-point của input (cần cho ESP32)
in_scale, in_zero_point = input_details[0]['quantization']
out_scale, out_zero_point = output_details[0]['quantization']
print(f'\nInput  quantization: scale={in_scale:.6f}, zero_point={in_zero_point}')
print(f'Output quantization: scale={out_scale:.6f}, zero_point={out_zero_point}')
# ── Chạy inference TFLite trên toàn bộ tập test ──────────────────────────
y_pred_tflite = []

for i in range(len(X_test)):
    sample_f32 = X_test[i:i+1].astype(np.float32)   # (1, 150, 3, 1)

    # Chuyển float32 → int8 đúng công thức: q = round(x/scale + zero_point), rồi clip [-128, 127]
    sample_int8 = np.round(sample_f32 / in_scale + in_zero_point)
    sample_int8 = np.clip(sample_int8, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], sample_int8)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(output_details[0]['index'])  # (1, 2) int8

    # int8 → float32 score
    output_f32 = (output_int8.astype(np.float32) - out_zero_point) * out_scale
    pred_label = np.argmax(output_f32, axis=1)[0]
    y_pred_tflite.append(pred_label)

y_pred_tflite = np.array(y_pred_tflite)

acc_tfl  = accuracy_score(y_test, y_pred_tflite)
prec_tfl = precision_score(y_test, y_pred_tflite, zero_division=0)
rec_tfl  = recall_score(y_test, y_pred_tflite, zero_division=0)
f1_tfl   = f1_score(y_test, y_pred_tflite, zero_division=0)
cm_tfl   = confusion_matrix(y_test, y_pred_tflite, labels=[0, 1])
tn_tfl, fp_tfl, fn_tfl, tp_tfl = cm_tfl.ravel()
spec_tfl = tn_tfl / (tn_tfl + fp_tfl) if (tn_tfl + fp_tfl) > 0 else 0.0

print('=' * 50)
print('  KẾT QUẢ TFLite INT8 MODEL (test set)')
print('=' * 50)
print(f'  Accuracy    : {acc_tfl*100:.2f}%')
print(f'  Precision   : {prec_tfl*100:.2f}%')
print(f'  Sensitivity : {rec_tfl*100:.2f}%')
print(f'  Specificity : {spec_tfl*100:.2f}%')
print(f'  F1-Score    : {f1_tfl:.4f}')
print('=' * 50)


# ─── Đánh giá breakdown theo từng vị trí trên tập test ──────────────────────
print()
print('=' * 58)
print('  KẾT QUẢ TFLite INT8 — BREAKDOWN THEO VỊ TRÍ (test set)')
print('=' * 58)
print(f'  {"Vị trí":<8}  {"Acc":>7}  {"Precision":>9}  {"Recall":>8}  {"Spec":>7}  {"F1":>7}  {"N":>5}')
print(f'  {"-"*8}  {"-"*7}  {"-"*9}  {"-"*8}  {"-"*7}  {"-"*7}  {"-"*5}')

for pos in POSITIONS:
    mask = pos_test == pos
    if mask.sum() == 0:
        continue
    y_true_pos = y_test[mask]
    y_pred_pos = y_pred_tflite[mask]

    a   = accuracy_score(y_true_pos, y_pred_pos)
    p   = precision_score(y_true_pos, y_pred_pos, zero_division=0)
    r   = recall_score(y_true_pos, y_pred_pos, zero_division=0)
    f   = f1_score(y_true_pos, y_pred_pos, zero_division=0)
    cm_ = confusion_matrix(y_true_pos, y_pred_pos, labels=[0, 1])
    tn_, fp_, fn_, tp_ = cm_.ravel()
    sp  = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0.0

    print(f'  {pos:<8}  {a*100:>6.2f}%  {p*100:>8.2f}%  {r*100:>7.2f}%  {sp*100:>6.2f}%  {f:>7.4f}  {mask.sum():>5}')

print('=' * 58)

# ─── 8.4  Test TFLite int8 model trên tập KFall (cross-dataset) ─────────────
# KFall: 32 subject trẻ, 21 ADL (T01-T21) + 15 Fall (T22-T36), sensor đặt ở
# lưng dưới (low back), tần số gốc 100 Hz, file .csv: 11 cột =
# [TimeStamp, FrameCounter, AccX, AccY, AccZ (g), GyroX, GyroY, GyroZ (°/s),
#  EulerX, EulerY, EulerZ (°)].
# Cấu trúc thư mục: sensor_data/<subject_id>/<file>.csv
# Tên file dạng "S06T01R01.csv" -> Subject 06, Task 01, Trial 01.
#
# Task ID theo paper gốc KFall (Yu et al., 2021, Frontiers in Aging Neuroscience):
#   ADL : D01-D21 → file T01-T21  (Task ID 1-21)
#   Fall: F01-F15 → file T22-T36  (Task ID 22-36)
KFALL_ROOT = '/kaggle/input/datasets/usmanabbasi2002/kfall-dataset/KFall Dataset/KFall Dataset/sensor_data'

KFALL_ORIG_FS       = 100         # Hz - tần số gốc KFall
KFALL_DOWNSAMPLE_N  = KFALL_ORIG_FS // TARGET_FS   # 100/25 = 4
KFALL_FALL_TASK_MIN = 22          # Task ID nhỏ nhất được coi là Fall (F01 = T22)
KFALL_FALL_TASK_MAX = 36          # Task ID lớn nhất được coi là Fall (F15 = T36)

if not os.path.isdir(KFALL_ROOT):
    candidates = glob.glob('/kaggle/input/**/sensor_data', recursive=True) + \
                 glob.glob('/kaggle/input/**/*KFall*', recursive=True)
    if candidates:
        KFALL_ROOT = candidates[0]
        print('Đã tự tìm thấy KFALL_ROOT:', KFALL_ROOT)
    else:
        print('Không tìm thấy KFall trong /kaggle/input. Hãy kiểm tra dataset đã Add chưa.')

print('KFALL_ROOT =', KFALL_ROOT)


def parse_kfall_task_id(filepath):
    """Lấy Task ID (int) từ tên file dạng S06T01R01.csv -> 1"""
    fname = os.path.basename(filepath)
    m = re.search(r'T(\d+)', fname)
    return int(m.group(1)) if m else None


def parse_kfall_file(filepath, debug=False):
    """
    Đọc 1 file .csv KFall, trả về mảng accelerometer (N, 3) đơn vị g.
    Cột accel trong KFall đã có sẵn đơn vị g, không cần convert raw counts.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None

    if debug:
        print('Các cột trong file mẫu:', list(df.columns))

    # Tên cột accel có thể khác nhau giữa các bản KFall, nên dò theo từ khóa
    acc_cols = [c for c in df.columns if re.search(r'Acc[_]?[XYZ]', str(c), re.IGNORECASE)]
    if len(acc_cols) < 3:
        # fallback: nếu không khớp tên cột, dùng vị trí cột 2,3,4 (sau Timestamp, FrameCounter)
        if df.shape[1] >= 5:
            accel = df.iloc[:, 2:5].to_numpy(dtype=np.float32)
        else:
            return None
    else:
        accel = df[acc_cols[:3]].to_numpy(dtype=np.float32)

    return accel  # (N, 3) đơn vị g


# ── In thử header của 1 file mẫu để kiểm tra tên cột có khớp quy ước trên không ──
_sample_files = glob.glob(os.path.join(KFALL_ROOT, '**', '*.csv'), recursive=True)[:1]
if _sample_files:
    parse_kfall_file(_sample_files[0], debug=True)


def load_kfall(root):
    """
    Quét toàn bộ thư mục KFall, gán nhãn theo Task ID, downsample 100→25 Hz,
    cắt cửa sổ 150 mẫu bằng cùng hàm extract_window() của SisFall, rồi
    normalize bằng đúng x_min_global/x_max_global đã học từ tập train SisFall.
    """
    X, y, groups = [], [], []

    csv_files = glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)
    print(f'Tìm thấy {len(csv_files)} file .csv trong KFall')

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f'Không tìm thấy file .csv trong KFALL_ROOT={root}. '
            'Hãy kiểm tra đường dẫn dataset KFall.'
        )

    skipped = 0
    for fp in csv_files:
        task_id = parse_kfall_task_id(fp)
        if task_id is None:
            skipped += 1
            continue

        label = 1 if (KFALL_FALL_TASK_MIN <= task_id <= KFALL_FALL_TASK_MAX) else 0

        accel = parse_kfall_file(fp)
        if accel is None or len(accel) < KFALL_DOWNSAMPLE_N * 20:
            skipped += 1
            continue

        accel_ds = downsample(accel, KFALL_DOWNSAMPLE_N)
        window = extract_window(accel_ds)   # (150, 3), tái dùng hàm đã có

        # Subject id lấy từ tên thư mục cha, ví dụ sensor_data/SA06/... -> SA06
        subject_id = os.path.basename(os.path.dirname(fp))

        X.append(window)
        y.append(label)
        groups.append(subject_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups)

    if len(X) == 0:
        raise ValueError('Không load được mẫu hợp lệ nào từ KFall.')

    print(f'Load xong: {len(X)} mẫu, bỏ qua {skipped} file')
    print(f'  ADL  (0): {np.sum(y == 0)}')
    print(f'  Fall (1): {np.sum(y == 1)}')
    print(f'  Số subject: {len(np.unique(groups))}')

    return X, y, groups


X_kfall_raw, y_kfall, groups_kfall = load_kfall(KFALL_ROOT)
X_kfall_cnn = X_kfall_raw[..., np.newaxis]   # (N, 150, 3, 1)

# Normalize bằng ĐÚNG min/max đã học từ tập train SisFall (không tính lại mới)
X_kfall_norm = normalize_accel(X_kfall_cnn, x_min_global, x_max_global).astype(np.float32)
X_kfall_norm = np.clip(X_kfall_norm, -1.0, 1.0)   # phòng giá trị KFall vượt range train

print(f'X_kfall_norm shape: {X_kfall_norm.shape}')

# ── Quantize sang int8 và chạy qua TFLite Interpreter (tflite_path đã convert ở B7) ──
y_pred_kfall = []
for i in range(len(X_kfall_norm)):
    sample_f32 = X_kfall_norm[i:i+1].astype(np.float32)

    sample_int8 = np.round(sample_f32 / in_scale + in_zero_point)
    sample_int8 = np.clip(sample_int8, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], sample_int8)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(output_details[0]['index'])

    output_f32 = (output_int8.astype(np.float32) - out_zero_point) * out_scale
    pred_label = np.argmax(output_f32, axis=1)[0]
    y_pred_kfall.append(pred_label)

y_pred_kfall = np.array(y_pred_kfall)

acc_kf  = accuracy_score(y_kfall, y_pred_kfall)
prec_kf = precision_score(y_kfall, y_pred_kfall, zero_division=0)
rec_kf  = recall_score(y_kfall, y_pred_kfall, zero_division=0)
f1_kf   = f1_score(y_kfall, y_pred_kfall, zero_division=0)
cm_kf   = confusion_matrix(y_kfall, y_pred_kfall, labels=[0, 1])
tn_kf, fp_kf, fn_kf, tp_kf = cm_kf.ravel()
spec_kf = tn_kf / (tn_kf + fp_kf) if (tn_kf + fp_kf) > 0 else 0.0

print('=' * 50)
print('  KẾT QUẢ TFLite INT8 MODEL trên tập KFall')
print('=' * 50)
print(f'  Accuracy    : {acc_kf*100:.2f}%')
print(f'  Precision   : {prec_kf*100:.2f}%')
print(f'  Sensitivity : {rec_kf*100:.2f}%')
print(f'  Specificity : {spec_kf*100:.2f}%')
print(f'  F1-Score    : {f1_kf:.4f}')
print('=' * 50)
print()
print(classification_report(y_kfall, y_pred_kfall, target_names=['ADL', 'Fall'], zero_division=0))

# ── Confusion matrix KFall ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_kf, cmap='Greens')
plt.colorbar(im)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm_kf[i,j]}', ha='center', va='center',
                color='white' if cm_kf[i,j] > cm_kf.max()/2 else 'black', fontsize=14)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['ADL','Fall']); ax.set_yticklabels(['ADL','Fall'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix — TFLite Int8 trên KFall (cross-dataset)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_kfall_int8.png'), dpi=150)
plt.show()

# ───────────────────────────────────────────────────────────────────────────
# 12. Vẽ biểu đồ so sánh Float32 và Int8 cho báo cáo
# ───────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import os

# Gom nhóm các chỉ số thành mảng (Đưa F1-score về thang 100% để vẽ cùng trục)
metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score']

float32_scores = [acc * 100, prec * 100, rec * 100, spec * 100, f1 * 100]
int8_scores    = [acc_tfl * 100, prec_tfl * 100, rec_tfl * 100, spec_tfl * 100, f1_tfl * 100]

x = np.arange(len(metrics))  # Vị trí các nhóm cột
width = 0.35                 # Độ rộng của cột

fig, ax = plt.subplots(figsize=(10, 6))

# Vẽ 2 nhóm cột
rects1 = ax.bar(x - width/2, float32_scores, width, label='Float32 (QAT)', color='#4C72B0')
rects2 = ax.bar(x + width/2, int8_scores, width, label='Int8 (TFLite)', color='#DD8452')

# Trang trí trục và tiêu đề
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Comparison of Model Performance: Float32 vs. Int8 (Accelerometer only)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 110) # Để khoảng trống phía trên cho dễ nhìn số liệu
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Hàm để ghi số liệu lên đỉnh mỗi cột
def autolabel(rects):
    """Đính kèm text hiển thị giá trị lên trên mỗi cột"""
    for rect in rects:
        height = rect.get_height()
        # Định dạng 2 chữ số thập phân
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Dịch lên 3 points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# Lưu hình ảnh độ phân giải cao để chèn vào báo cáo Word/PDF
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'comparison_float32_vs_int8.png')
plt.savefig(chart_path, dpi=300)
plt.show()

print(f'\nĐã lưu biểu đồ so sánh tại: {chart_path}')
def tflite_to_c_array(tflite_bytes, var_name='fd_cnn_multipos_model'):
    """Chuyển .tflite binary thành C/C++ header array."""
    lines = []
    lines.append(f'// Auto-generated from fd_cnn_multipos_int8.tflite')
    lines.append(f'// Trained on SisFall with 4 sensor positions: front, right, left, back')
    lines.append(f'// Model size: {len(tflite_bytes)} bytes')
    lines.append(f'')
    lines.append(f'#pragma once')
    lines.append(f'#include <stdint.h>')
    lines.append(f'')
    lines.append(f'// Normalization params for Accelerometer (Axes 0, 1, 2)')
    lines.append(f'#define NORM_ACCEL_MIN  ({x_min_global:.6f}f)')
    lines.append(f'#define NORM_ACCEL_MAX  ({x_max_global:.6f}f)')
    lines.append(f'')
    lines.append(f'// TFLite quantization params')
    lines.append(f'// q = round(x_norm / INPUT_SCALE + INPUT_ZERO_POINT), then clip to [-128, 127]')
    lines.append(f'#define INPUT_SCALE       ({in_scale:.8f}f)')
    lines.append(f'#define INPUT_ZERO_POINT  ({in_zero_point})')
    lines.append(f'#define OUTPUT_SCALE      ({out_scale:.8f}f)')
    lines.append(f'#define OUTPUT_ZERO_POINT ({out_zero_point})')
    lines.append(f'')
    lines.append(f'// Window: {WINDOW_LEN} samples × 3 axes (accelerometer) × 1 channel')
    lines.append(f'#define MODEL_INPUT_LEN   ({WINDOW_LEN})')
    lines.append(f'#define MODEL_INPUT_AXES  (3)')
    lines.append(f'#define MODEL_NUM_CLASSES (2)  // 0=ADL, 1=Fall')
    lines.append(f'')
    lines.append(f'#ifdef __cplusplus')
    lines.append(f'alignas(8) const unsigned char {var_name}[] = {{')
    lines.append(f'#else')
    lines.append(f'const unsigned char {var_name}[] __attribute__((aligned(8))) = {{')
    lines.append(f'#endif')
    hex_vals = [f'0x{b:02x}' for b in tflite_bytes]
    for chunk_start in range(0, len(hex_vals), 16):
        chunk = hex_vals[chunk_start:chunk_start+16]
        lines.append('  ' + ', '.join(chunk) + ',')
    lines.append('};')
    lines.append(f'const int {var_name}_len = {len(tflite_bytes)};')
    return '\n'.join(lines)


c_header = tflite_to_c_array(tflite_int8_model)
header_path = os.path.join(OUTPUT_DIR, 'fd_cnn_multipos_model.h')
with open(header_path, 'w') as f:
    f.write(c_header)

header_kb = os.path.getsize(header_path) / 1024
print(f'C header đã lưu: {header_path}  ({header_kb:.1f} KB)')

print('=' * 65)
print('              TÓM TẮT KẾT QUẢ — MULTI-POSITION')
print('=' * 65)
print(f'  Tập dữ liệu   : SisFall (down-sampled 200→25 Hz)')
print(f'  Vị trí train  : {POSITIONS}')
print(f'  Model         : FD-CNN Lightweight (Khawnuan et al., 2023)')
print(f'  Phương pháp   : QAT (Quantization-Aware Training)')
print(f'  Quantization  : int8 (weight + activation + I/O)')
print()
print(f'  [Float32 QAT model - test set tổng hợp]')
print(f'    Accuracy    : {acc*100:.2f}%')
print(f'    F1-Score    : {f1:.4f}')
print()
print(f'  [TFLite int8 model - test set tổng hợp]')
print(f'    Accuracy    : {acc_tfl*100:.2f}%')
print(f'    F1-Score    : {f1_tfl:.4f}')
print(f'    Model size  : {size_kb:.1f} KB')
print('=' * 65)
print()
print('Output files:')
print(f'  /kaggle/working/fd_cnn_multipos_int8.tflite   ← deploy TFLite Micro')
print(f'  /kaggle/working/fd_cnn_multipos_model.h       ← C header cho ESP32')
print(f'  /kaggle/working/best_qat_model.weights.h5     ← QAT weights')
print()
print('Tham số cần thiết cho ESP32:')
print(f'  WINDOW_LEN       = {WINDOW_LEN}  (samples)')
print(f'  ORIG_FS / DS_FS  = {ORIG_FS}/{TARGET_FS} Hz  → bước lấy mẫu: {DOWNSAMPLE_N}')
print(f'  NORM_ACCEL_MIN   = {x_min_global:.6f} g')
print(f'  NORM_ACCEL_MAX   = {x_max_global:.6f} g')
print(f'  INPUT_SCALE      = {in_scale:.8f}')
print(f'  INPUT_ZERO_POINT = {in_zero_point}')
print()
print('Lưu ý ESP32: áp đúng ma trận xoay cho vị trí gắn thực tế trước khi')
print('  đưa dữ liệu vào model. Xem POSITION_TRANSFORMS trong notebook.')

# ───────────────────────────────────────────────────────────────────────────
# 13. Nén toàn bộ file kết quả thành file ZIP để tải về
# ───────────────────────────────────────────────────────────────────────────
import os
import zipfile

def create_project_zip(output_dir, zip_filename='fd_cnn_esp32_project_da_vitri_dat.zip'):
    """
    Gom các file kết quả (model, weights, header, đồ thị) thành 1 file zip.
    """
    zip_path = os.path.join(output_dir, zip_filename)
    
    # Chỉ chọn nén các file có đuôi này để tránh nén nhầm file rác của hệ thống
    valid_extensions = ('.h5', '.tflite', '.h', '.png')
    
    print(f"Đang tạo file ZIP: {zip_filename} ...")
    
    # Mở file zip ở chế độ ghi (w) và áp dụng thuật toán nén (ZIP_DEFLATED)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                # Bỏ qua chính file zip đang được tạo để tránh lỗi lặp vòng
                if file == zip_filename:
                    continue
                    
                # Nếu file có đuôi hợp lệ thì thêm vào zip
                if file.endswith(valid_extensions):
                    file_path = os.path.join(root, file)
                    
                    # Lấy tên tương đối để bên trong file zip không bị vướng đường dẫn thư mục gốc
                    arcname = os.path.relpath(file_path, output_dir) 
                    zipf.write(file_path, arcname)
                    print(f" ├── Đã thêm: {arcname}")
                    
    # Lấy kích thước file zip
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    return zip_path, size_mb

# Thực thi hàm
zip_filepath, zip_size = create_project_zip(OUTPUT_DIR)

print('=' * 60)
print(' ✅ HOÀN TẤT NÉN FILE')
print('=' * 60)
print(f' 📁 Đường dẫn : {zip_filepath}')
print(f' 💾 Dung lượng : {zip_size:.2f} MB')
print('\nBây giờ bạn có thể mở thanh menu bên phải của Kaggle (phần Data -> Output) ')
print('hoặc Colab (phần biểu tượng thư mục) để tải file "fd_cnn_esp32_project.zip" về máy!')