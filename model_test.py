import torch
from collections import defaultdict

from model import *
from dataset import *

length_total = defaultdict(int)
length_correct = defaultdict(int)

def model_test(model):
    model.eval()
    with torch.no_grad():
        for i in range(50000):
            if i % 5000 == 0:
                print(f"{i}번 test")
            if i == (50000-1):
                print(f"{i}번 test")
            # batch_size는 1로 지정
            sequence, x, label  = generate_data(1, 20, 1)

            x = torch.tensor(x, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)

            output = model(x)
            length_total[sequence.size] += 1
            if torch.all(output.argmax(dim=-1) == label):
                length_correct[sequence.size] += 1




            