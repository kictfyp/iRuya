import cv2
import numpy as np
import time
import psutil
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# ====================== CONFIG ======================
IMAGE_FOLDER = r"C:\Users\sarif\OneDrive\Main Folder for all documents\MY MCS\Thesis Salman Shah\Salman-Thesis\EDGE DETECTION LIVE PHOTOS"
TOTAL_IMAGES = 500
TEST_SIZE = 100
RESIZE_TO = (640, 480)
REPEATS = 3
random.seed(42)

# ====================== LOAD IMAGES ======================
def load_images(folder, limit=TOTAL_IMAGES):
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(filenames)} images.")
    
    for i, filename in enumerate(filenames[:limit]):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, RESIZE_TO)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append((filename, gray))
    return images

all_images = load_images(IMAGE_FOLDER)
random.shuffle(all_images)
test_images = all_images[-TEST_SIZE:]   # Last 100 as test

print(f"Test set: {len(test_images)} images\n")

# ====================== ALGORITHMS ======================
def process_canny(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    return cv2.Canny(blurred, 50, 150)

def process_sobel(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    _, edges = cv2.threshold(sobel.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
    return edges

def process_prewitt(gray):
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
    prewittx = cv2.filter2D(gray.astype(np.float32), -1, kernelx)
    prewitty = cv2.filter2D(gray.astype(np.float32), -1, kernely)
    prewitt = cv2.magnitude(prewittx, prewitty)
    _, edges = cv2.threshold(prewitt.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
    return edges

# ====================== BENCHMARK ======================
def benchmark(images_set, algorithm_func, name):
    total_time = 0.0
    cpu_readings = []
    
    print(f"Running {name} ...")
    for repeat in range(REPEATS):
        for i, (filename, gray) in enumerate(images_set):
            start = time.perf_counter()
            _ = algorithm_func(gray)
            total_time += (time.perf_counter() - start)
            
            cpu_readings.append(psutil.cpu_percent(interval=0.05))
            
    num_measurements = len(images_set) * REPEATS
    avg_time_ms = (total_time / num_measurements) * 1000
    fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0
    avg_cpu = np.mean(cpu_readings)
    
    return {
        'Algorithm': name,
        'FPS': round(fps, 2),
        'Avg_Time_ms': round(avg_time_ms, 3),
        'CPU_Usage_%': round(avg_cpu, 2)
    }

# ====================== RUN ======================
results = []
results.append(benchmark(test_images, process_canny, "Canny"))
results.append(benchmark(test_images, process_sobel, "Sobel"))
results.append(benchmark(test_images, process_prewitt, "Prewitt"))

df = pd.DataFrame(results)
print("\n" + "="*85)
print("FINAL RESULTS")
print("="*85)
print(df.to_string(index=False))

df.to_csv("edge_detection_results.csv", index=False)
print("\n✅ Results saved to edge_detection_results.csv")

# ==================== MULTIPLE FIGURES ====================

# Figure 1: FPS Comparison
plt.figure(figsize=(8,5))
plt.bar(df['Algorithm'], df['FPS'], color=['blue', 'green', 'orange'])
plt.title('Frame Rate (FPS) Comparison')
plt.ylabel('FPS')
plt.grid(axis='y')
plt.savefig('Figure_FPS_Comparison.png')
plt.close()

# Figure 2: Average Time
plt.figure(figsize=(8,5))
plt.bar(df['Algorithm'], df['Avg_Time_ms'], color=['blue', 'green', 'orange'])
plt.title('Average Processing Time per Frame')
plt.ylabel('Time (ms)')
plt.grid(axis='y')
plt.savefig('Figure_Avg_Time.png')
plt.close()

# Figure 3: CPU Usage
plt.figure(figsize=(8,5))
plt.bar(df['Algorithm'], df['CPU_Usage_%'], color=['blue', 'green', 'orange'])
plt.title('CPU Usage Comparison')
plt.ylabel('CPU Usage (%)')
plt.grid(axis='y')
plt.savefig('Figure_CPU_Usage.png')
plt.close()

# Figure 4: Combined
df.plot(x='Algorithm', y=['FPS', 'Avg_Time_ms', 'CPU_Usage_%'], kind='bar', figsize=(10,6))
plt.title('Overall Performance Comparison')
plt.ylabel('Value')
plt.grid(axis='y')
plt.savefig('Figure_Overall_Performance.png')
plt.close()

print("\n✅ All figures saved successfully!")
print("   - Figure_FPS_Comparison.png")
print("   - Figure_Avg_Time.png")
print("   - Figure_CPU_Usage.png")
print("   - Figure_Overall_Performance.png")