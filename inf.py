import numpy as np

def inf():
    """1000 times 1000 matrix mult """
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    return np.dot(a, b)

if __name__ == "__main__":
    while True:
        inf()