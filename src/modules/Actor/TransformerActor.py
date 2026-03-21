import torch
import torch.nn.functional as F

class TransformerActor(torch.nn.Module):
    def __init__(self, args):
        super(TransformerActor, self).__init__()
        self.n_agents = args.num_agents
        self.device = args.device
        self.state_size = args.state_size
        self.action_size = args.action_size

        self.embedding_dim = args.embadding_dim

        # 상태를 임베딩 차원으로 변환
        self.fc1 = torch.nn.Linear(self.state_size, self.embedding_dim).to(self.device)

        # 포지셔널 인코딩 추가
        self.positional_encoding = PositionalEncoding(self.embedding_dim).to(self.device)

        # Transformer 인코더 레이어 생성
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6).to(self.device)

        # 액션을 출력하기 위한 완전연결층
        self.fc2 = torch.nn.Linear(self.embedding_dim, self.action_size).to(self.device)

    def forward(self, inputs):
        # inputs: (batch_size, sequence_length, state_size)
        x = self.fc1(inputs)  # x: (batch_size, sequence_length, embedding_dim)
        x = self.positional_encoding(x)  # 포지셔널 인코딩 적용

        # Transformer는 (sequence_length, batch_size, embedding_dim) 형태의 입력을 기대하므로 변환
        x = x.permute(1, 0, 2)

        # Transformer 인코더 통과
        output = self.transformer_encoder(x)

        # 다시 (batch_size, sequence_length, embedding_dim) 형태로 변환
        output = output.permute(1, 0, 2)

        # 마지막 시퀀스의 출력만 사용하여 액션 예측
        q = self.fc2(output[:, -1, :])  # output[:, -1, :]는 마지막 타임스텝의 출력

        return F.softmax(q, dim=-1)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 포지셔널 인코딩 계산
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스에 sin 적용
        pe[0, :, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스에 cos 적용

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, sequence_length, embedding_dim)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x
