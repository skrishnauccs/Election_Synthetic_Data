# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# x = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Naive Byes', 'XGBoost']
# y1 = [0.8755, 0.8635, 0.8772, 0.8665, 0.7397, 0.887, ]
# y2 = [0.8635, 0.8687, 0.8747, 0.8402, 0.8657, 0.8655]
# y3 = [0.8755, 0.8842, 0.8777, 0.8665, 0.7577, 0.884]
#
# plt.plot(x, y1, marker='o', label="Benchmark Results")
# plt.plot(x, y2, marker='o', label="Synthetic Data Results")
# plt.plot(x, y3, marker='o', label="Synthetic Data Results after HT")
# plt.title("Accuracy Comparison of Real Data and Synthetic Data")
# plt.legend()
# plt.show()

# Recall
# x = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Naive Byes', 'XGBoost']
# y1 = [
# 0.8142,
# 0.8237,
# 0.8171,
# 0.8338,
# 0.7703,
# 0.83210
# ]
# y2 = [
# 0.8029,
# 0.8101,
# 0.8131,
# 0.7828,
# 0.6111,
# 0.8094
# ]
# y3 = [
# 0.8139,
# 0.8274,
# 0.8177,
# 0.8338,
# 0.7741,
# 0.8273
# ]
#
# plt.plot(x, y1, marker='o', label="Benchmark Results")
# plt.plot(x, y2, marker='o', label="Synthetic Data Results")
# plt.plot(x, y3, marker='o', label="Synthetic Data Results after HT")
# plt.title("Recall Comparison of Real Data and Synthetic Data")
# plt.legend()
# plt.show()

# Precision
# x = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Naive Byes', 'XGBoost']
# y1 = [
# 0.8115,
# 0.8068,
# 0.7885,
# 0.6700,
# 0.8524,
# 0.8142
# ]
# y2 = [
# 0.8244,
# 0.8345,
# 0.7952,
# 0.8174,
# 0.6666,
# 0.8448,
# ]
# y3 = [
# 0.8085,
# 0.8119,
# 0.7915,
# 0.6700,
# 0.8573,
# 0.80336,
# ]
#
# plt.plot(x, y1, marker='o', label="Benchmark Results")
# plt.plot(x, y2, marker='o', label="Synthetic Data Results")
# plt.plot(x, y3, marker='o', label="Synthetic Data Results after HT")
# plt.title("Precision Comparison of Real Data and Synthetic Data")
# plt.legend()
# plt.show()

# f1-socre

x = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Naive Byes', 'XGBoost']
y1 = [
0.8128,
0.8143,
0.8003,
0.6438,
0.7512,
0.8222
]
y2 = [
0.8119,
0.8202,
0.8032,
0.7944,
0.6363,
0.8226
]
y3 = [
0.8111,
0.8189,
0.8025,
0.6438,
0.7627,
0.8135
]

plt.plot(x, y1, marker='o', label="Benchmark Results")
plt.plot(x, y2, marker='o', label="Synthetic Data Results")
plt.plot(x, y3, marker='o', label="Synthetic Data Results after HT")
plt.title("F1-score Comparison of Real Data and Synthetic Data")
plt.legend()
plt.show()