import matplotlib.pyplot as plt
import numpy as np

x = np.asarray([12, 8, 4, 2, 0])
y = np.asarray([0.9386, 0.7152, 0.3214, 0.1882, 0.0512])

plt.plot(x, y, linestyle='--', marker='o', color='y', label='bert-base Query Encoder')
plt.xlim(max(x), min(x))
plt.margins(y=0.1)
plt.ylabel('Top-10 Accuracy (%)', fontsize=16)
plt.xlabel('N: # of Query Encoder Layers', fontsize=16)

plt.legend()

plt.show()