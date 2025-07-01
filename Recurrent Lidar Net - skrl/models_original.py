import torch
import torch.nn as nn
import torch.nn.functional as F

# LSTM-based network for PPO/A2C (one LSTM layer + actor and critic heads)
class BasicLSTMNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(BasicLSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=False)
        self.actor = nn.Linear(hidden_dim, action_dim)   # outputs mean action (for continuous) or logits (for discrete)
        self.critic = nn.Linear(hidden_dim, 1)           # outputs state value
        # Learnable log_std for Gaussian policy (used if continuous actions)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, x):
        # If input has no time dimension (batch, obs_dim), add seq_len = 1
        if x.dim() == 2:
            x = x.unsqueeze(0)  # shape -> (1, batch, obs_dim)
        lstm_out, _ = self.lstm(x)                # lstm_out: (seq_len, batch, hidden_dim)
        feat = lstm_out[-1]                       # take output of last timestep
        action_mean = self.actor(feat)
        state_value = self.critic(feat)
        return action_mean, state_value

####################################################################################

# Spatial+Temporal network: MLP feature extractor + LSTM for PPO/A2C
class SpatioTemporalRLNNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(SpatioTemporalRLNNet, self).__init__()
        self.hidden_dim = hidden_dim
        # MLP feature extractor for spatial features (e.g., LiDAR)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )
        # LSTM for temporal sequence processing
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=False)
        # Policy and value heads
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        seq_len, batch_size, _ = x.shape
        # Apply feature extractor to each timestep of the sequence
        xf = x.reshape(seq_len * batch_size, -1)
        feat = self.feature_extractor(xf)
        feat = feat.reshape(seq_len, batch_size, -1)
        lstm_out, _ = self.lstm(feat)
        feat_t = lstm_out[-1]  # last timestep features
        action_mean = self.actor(feat_t)
        state_value = self.critic(feat_t)
        return action_mean, state_value

####################################################################################

# Transformer-based network for PPO/A2C (self-attention over LiDAR beams + state)
class TransformerNet(nn.Module):
    def __init__(self, obs_dim, action_dim, n_heads=4, n_layers=2, embed_dim=64):
        super(TransformerNet, self).__init__()
        # Assume obs consists of [speed, LiDAR_range_values...]
        self.state_dim = 1  # first element is speed
        self.lidar_dim = obs_dim - self.state_dim
        # Split LiDAR into fixed number of tokens (e.g., 6 tokens)
        self.lidar_tokens = 6
        if self.lidar_dim > 0:
            self.points_per_token = max(1, self.lidar_dim // self.lidar_tokens)
            self.lidar_tokens = self.lidar_dim // self.points_per_token
        else:
            self.points_per_token = 0
            self.lidar_tokens = 0
        self.num_tokens = self.lidar_tokens + (1 if self.state_dim > 0 else 0)
        # Embedding layers for state and LiDAR tokens
        if self.state_dim > 0:
            self.state_embedding = nn.Linear(self.state_dim, embed_dim)
        if self.lidar_tokens > 0:
            self.lidar_embedding = nn.Linear(self.points_per_token, embed_dim)
        # Positional embedding for tokens
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=4*embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Final fully-connected layer after transformer
        self.fc = nn.Sequential(
            nn.Linear(self.num_tokens * embed_dim, 256),
            nn.ReLU()
        )
        # Policy and value output layers
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, obs_batch):
        batch_size = obs_batch.shape[0]
        tokens = []
        # State token
        if self.state_dim > 0:
            state_in = obs_batch[:, :self.state_dim]           # shape (batch, 1)
            state_tok = self.state_embedding(state_in)         # shape (batch, embed_dim)
            tokens.append(state_tok.unsqueeze(1))              # (batch, 1, embed_dim)
        # LiDAR tokens
        if self.lidar_tokens > 0:
            lidar_in = obs_batch[:, self.state_dim:]           # shape (batch, lidar_dim)
            for i in range(self.lidar_tokens):
                start = i * self.points_per_token
                end = start + self.points_per_token
                if end > lidar_in.shape[1]:
                    chunk = F.pad(lidar_in[:, start:], (0, end - lidar_in.shape[1]))
                else:
                    chunk = lidar_in[:, start:end]
                tok = self.lidar_embedding(chunk)              # (batch, embed_dim) for this token
                tokens.append(tok.unsqueeze(1))                # (batch, 1, embed_dim)
        # Concatenate tokens and add positional encoding
        token_seq = torch.cat(tokens, dim=1) if tokens else torch.zeros((batch_size, 0))
        token_seq = token_seq + self.pos_embedding[:, :token_seq.size(1), :]
        # Forward through transformer encoder
        trans_out = self.transformer(token_seq)                # (batch, num_tokens, embed_dim)
        # Flatten token outputs and apply final FC
        flat_out = trans_out.reshape(batch_size, -1)
        feat = self.fc(flat_out)
        # Output action mean and state value
        action_mean = self.actor(feat)
        state_value = self.critic(feat)
        return action_mean, state_value

####################################################################################

# Spatio-temporal Dueling Transformer for DQN
# input  shape:  (B, T, R, 2)   – T timesteps, R lidar ranges, 2 channels
# output shape:  (B, num_actions)  – state-action values  Q(s,a)
# Shryn's model
class SpatioTempDuelingTransformerNet(nn.Module):
    def __init__(self, seq_len: int, num_ranges: int, num_actions: int):
        super().__init__()
        # 1-D conv stack over (range, channel) for each time-step
        self.conv1 = nn.Conv1d(2, 16, 5, 2); self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 3, 1); self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 48, 4, 2); self.bn3 = nn.BatchNorm1d(48)
        self.conv4 = nn.Conv1d(48, 64, 3, 1); self.bn4 = nn.BatchNorm1d(64)

        # work out conv-feature length once so we can flatten later
        with torch.no_grad():
            dummy = torch.zeros(1, 2, num_ranges)
            feat = self._conv(dummy)
            conv_feat_size = feat.flatten(1).shape[1]

        self.proj = nn.Linear(conv_feat_size, 128)
        encoder = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder, 2)
        self.ln = nn.LayerNorm(128)

        # dueling heads
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv   = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_actions))

    # helper used once in __init__ and every forward
    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   (B, T, R, 2)
        B, T, R, _ = x.shape
        x = x.view(B*T, R, 2).permute(0, 2, 1)  # -> (B*T, 2, R)
        x = self._conv(x)                       # -> (B*T, C, R')
        x = x.flatten(1).view(B, T, -1)         # -> (B, T, C_flat)
        x = self.proj(x)                        # -> (B, T, 128)
        # Transformer expects (T, B, C)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        ctx = self.ln(x).mean(1)                # (B, 128)
        v = self.value(ctx)                     # (B, 1)
        a = self.adv(ctx)                       # (B, A)
        return v + a - a.mean(1, keepdim=True)  # Q(s,a)

####################################################################################
#From ChatGPT
# Dueling Transformer network for DQN (separate advantage and value streams)
class DuelingTransformerNet(nn.Module):
    def __init__(self, obs_dim, num_actions, n_heads=4, n_layers=2, embed_dim=64):
        super(DuelingTransformerNet, self).__init__()
        self.state_dim = 1
        self.lidar_dim = obs_dim - self.state_dim
        self.lidar_tokens = 6
        if self.lidar_dim > 0:
            self.points_per_token = max(1, self.lidar_dim // self.lidar_tokens)
            self.lidar_tokens = self.lidar_dim // self.points_per_token
        else:
            self.points_per_token = 0
            self.lidar_tokens = 0
        self.num_tokens = self.lidar_tokens + (1 if self.state_dim > 0 else 0)
        if self.state_dim > 0:
            self.state_embedding = nn.Linear(self.state_dim, embed_dim)
        if self.lidar_tokens > 0:
            self.lidar_embedding = nn.Linear(self.points_per_token, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=4*embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Separate fully-connected streams for advantage and value
        self.adv_fc = nn.Sequential(nn.Linear(self.num_tokens * embed_dim, 256), nn.ReLU())
        self.val_fc = nn.Sequential(nn.Linear(self.num_tokens * embed_dim, 256), nn.ReLU())
        self.advantage_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)
    def forward(self, obs_batch):
        batch_size = obs_batch.size(0)
        tokens = []
        if self.state_dim > 0:
            state_tok = self.state_embedding(obs_batch[:, :self.state_dim])
            tokens.append(state_tok.unsqueeze(1))
        if self.lidar_tokens > 0:
            lidar_in = obs_batch[:, self.state_dim:]
            for i in range(self.lidar_tokens):
                start = i * self.points_per_token
                end = start + self.points_per_token
                if end > lidar_in.shape[1]:
                    chunk = F.pad(lidar_in[:, start:], (0, end - lidar_in.shape[1]))
                else:
                    chunk = lidar_in[:, start:end]
                tok = self.lidar_embedding(chunk)
                tokens.append(tok.unsqueeze(1))
        token_seq = torch.cat(tokens, dim=1) if tokens else torch.zeros((batch_size, 0))
        token_seq = token_seq + self.pos_embedding[:, :token_seq.size(1), :]
        trans_out = self.transformer(token_seq)               # (batch, num_tokens, embed_dim)
        flat_out = trans_out.reshape(batch_size, -1)
        # Compute advantage and value streams
        adv_feat = self.adv_fc(flat_out)
        val_feat = self.val_fc(flat_out)
        advantages = self.advantage_head(adv_feat)            # (batch, num_actions)
        values = self.value_head(val_feat)                    # (batch, 1)
        # Combine to get Q-values (dueling DQN formula)
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values
