import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, dim_input, dim_recurrent, dim_output):
        super(RNN, self).__init__()

        """
        dim_input: C
        dim_recurrent: D
        dim_output: K
        """

        self.x2h = nn.Linear(dim_input, dim_recurrent)
        self.h2h = nn.Linear(dim_recurrent, dim_recurrent, bias=False)
        self.h2y = nn.Linear(dim_recurrent, dim_output)
        self.relu = nn.ReLU()

    def forward(self, x, h_t=None):

        """
        x: shape = (T, N, C)
        W_x: shape = (C, D)
        초기 h: shape = (1, N, D)
        W_h: shape = (D, D)

        => x X W_x: (T, N, C) X (C, D) = (T, N, D)
           (초기 h) X W_h: (1, N, D) X (D, D) = (1, N, D)  
           
           h: (T, N, D) + (1, N, D) = (T, N, D) broadcasting

        w_y: shape = (D, K)

        y: h X w_y
         = (T, N, D) X (D, K) = (T, N, K)  

        y: shape = (T, N, K)
        h: shape = (T, N, D)  
        """
        N = x.shape[1]
        D = self.h2h.weight.shape[0]


        # 초기 hidden state를 (1, N, D) shape의 0텐서로 설정
        if h_t is None:
            h_t = torch.zeros(1, N, D, dtype=torch.float32)

        h = []

        for i in range(x.shape[0]):
            h_t = self.x2h(x[i]) + self.h2h(h_t)
            h_t = self.relu(h_t)
            h.append(h_t)

        # 배열 h에 저장된 모든 hidden state들을 dim=0 방향으로 합치기
        all_h = torch.cat(h, dim=0)

        all_y = self.h2y(all_h)

        return all_y, all_h    



class Seq2Seq(nn.Module):
    def __init__(self, dim_input, dim_recurrent, dim_output):
        super(Seq2Seq, self).__init__()

        """
        dim_input: 입력 데이터 차원(C)
        dim_recurrent: hidden state 차원(D)
        dim_output: 디코더 출력 차원(K)
        """

        self.encoder = RNN(dim_input, dim_recurrent, dim_output)
        self.decoder = RNN(dim_input, dim_recurrent, dim_output)

    def forward(self, x):
        """
        x(각 시퀀스에 대한 one-hot Encoding 입력): shape = (T, N, C)
        y(각 시퀀스의 디코딩된 출력): shape = (T, N, K)
        """
        T, N, C = x.shape

        y = []

        # 인코더를 통해 hidden state를 받음, 인코더의 출력은 취급하지 않음
        _, enc_h = self.encoder(x)

        # 마지막 step에서의 hidden state를 지정
        h_t = enc_h[-1:]

        # <sos> 토큰 즉, 디코더 첫 step의 입력을 나타내는 start 토큰 설정
        sos = torch.zeros(1, N, C)
        sos[:, :, -2] = 1

        # 타임스텝 T 동안 디코더 반복 실행
        for _ in range(T):
            # 디코더에 sos와 hidden state h_t를 전달하여 출력 계산
            y_t, h_t = self.decoder(sos, h_t)
            # 출력을 리스트에 저장
            y.append(y_t)

            # 다음 step의 입력을 현재 출력의 One-hot Encoding으로 설정
            sos = F.one_hot(y_t.argmax(dim=-1), num_classes=12).float().unsqueeze(0)

        # 리스트 y에 저장된 모든 y들을 dim=0 방향으로 합치기
        y = torch.cat(y, dim=0)

        return y

        