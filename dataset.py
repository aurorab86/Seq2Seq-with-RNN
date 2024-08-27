import numpy as np

def generate_data(seq_length_min=1, seq_length_max=20, batch_size=10):
    T = np.random.randint(seq_length_min, seq_length_max + 1)
    x = np.random.randint(0, 10, (T, batch_size))
    one_hot_x = np.zeros((T + 1, batch_size, 12), dtype=np.float32)
    one_hot_x[np.arange(T).reshape(-1, 1), np.arange(batch_size), x] = 1
    one_hot_x[-1, :, -1] = 1
    ends = np.full(batch_size, 11).reshape(1, -1)
    y = np.concatenate([x[::-1], ends], axis=0)
    return x, one_hot_x, y