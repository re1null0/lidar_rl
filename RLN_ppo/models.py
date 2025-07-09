# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(model_type, obs_dim, action_dim, config):
    if model_type == "basic_lstm":
        return BasicLSTMNet(
            obs_dim,
            action_dim,
            hidden_dim=config.get("hidden_dim", 128)
        )
    elif model_type == "bilstm":
        return BiLSTMNet(
            obs_dim,
            action_dim,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 1)
        )
    elif model_type == "spatiotemporal_rln":
        return SpatioTemporalRLNNet(
            obs_dim,
            action_dim,
            hidden_dim=config.get("hidden_dim", 128)
        )
    elif model_type == "transformer":
        return TransformerNet(
            obs_dim,
            action_dim,
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            embed_dim=config.get("embed_dim", 64)
        )
    elif model_type == "dueling_transformer":
        print("DuelingTransformer selected for PPO; using TransformerNet instead.")
        return TransformerNet(
            obs_dim,
            action_dim,
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            embed_dim=config.get("embed_dim", 64)
        )
    elif model_type == "spatiotemp_dueling_transformer":
        seq_len    = config.get("seq_len", 4)
        num_ranges = config.get("num_ranges", 1080)
        return SpatioTempDuelingTransformerNet(seq_len, num_ranges, action_dim)
    else:
        # fallback
        return BasicLSTMNet(obs_dim, action_dim, hidden_dim=config.get("hidden_dim", 128))


class BiLSTMNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=True
        )
        # project concatenated forward+backward to hidden_dim
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)          # (seq_len, batch, hidden*2)
        last = lstm_out[-1]                 # (batch, hidden*2)
        feat = F.relu(self.proj(last))      # (batch, hidden)
        return self.actor(feat), self.critic(feat)
    

class BasicLSTMNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=False)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        feat = lstm_out[-1]
        return self.actor(feat), self.critic(feat)


class SpatioTemporalRLNNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU()
        )
        self.lstm   = nn.LSTM(hidden_dim, hidden_dim, batch_first=False)
        self.actor  = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        seq_len, batch, _ = x.shape
        xf = x.reshape(seq_len * batch, -1)
        feat = self.feature_extractor(xf).reshape(seq_len, batch, -1)
        lstm_out, _ = self.lstm(feat)
        last = lstm_out[-1]
        return self.actor(last), self.critic(last)


class TransformerNet(nn.Module):
    def __init__(self, obs_dim, action_dim, n_heads=4, n_layers=2, embed_dim=64):
        super().__init__()
        self.state_dim = 1
        self.lidar_dim = obs_dim - self.state_dim
        # compute tokens
        self.lidar_tokens    = 6 if self.lidar_dim>0 else 0
        self.points_per_token = max(1, self.lidar_dim//self.lidar_tokens) if self.lidar_tokens>0 else 0
        self.num_tokens      = self.lidar_tokens + (1 if self.state_dim>0 else 0)

        if self.state_dim>0:
            self.state_embedding = nn.Linear(self.state_dim, embed_dim)
        if self.lidar_tokens>0:
            self.lidar_embedding = nn.Linear(self.points_per_token, embed_dim)

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                             dim_feedforward=4*embed_dim,
                                             batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.fc = nn.Sequential(nn.Linear(self.num_tokens * embed_dim, 256),
                                 nn.ReLU())
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        B = obs.size(0)
        tokens = []
        if self.state_dim>0:
            st = self.state_embedding(obs[:, :self.state_dim]).unsqueeze(1)
            tokens.append(st)
        if self.lidar_tokens>0:
            ld = obs[:, self.state_dim:]
            for i in range(self.lidar_tokens):
                start = i*self.points_per_token
                chunk = ld[:, start:start+self.points_per_token]
                if chunk.size(1)<self.points_per_token:
                    chunk = F.pad(chunk, (0, self.points_per_token-chunk.size(1)))
                tokens.append(self.lidar_embedding(chunk).unsqueeze(1))
        seq = torch.cat(tokens, dim=1) if tokens else torch.zeros(B,0)
        seq = seq + self.pos_embedding[:, :seq.size(1), :]
        out = self.transformer(seq)
        flat = out.reshape(B, -1)
        feat = self.fc(flat)
        return self.actor(feat), self.critic(feat)


class SpatioTempDuelingTransformerNet(nn.Module):
    def __init__(self, seq_len, num_ranges, num_actions):
        super().__init__()
        # conv stack
        self.conv1 = nn.Conv1d(2,16,5,2); self.bn1=nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16,32,3,1); self.bn2=nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32,48,4,2); self.bn3=nn.BatchNorm1d(48)
        self.conv4 = nn.Conv1d(48,64,3,1); self.bn4=nn.BatchNorm1d(64)

        with torch.no_grad():
            dummy = torch.zeros(1,2,num_ranges)
            conv_feat = self._conv(dummy).flatten(1)
        self.proj = nn.Linear(conv_feat.shape[1], 128)
        enc = nn.TransformerEncoderLayer(d_model=128, nhead=4,
                                         dim_feedforward=256,
                                         batch_first=False)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.ln  = nn.LayerNorm(128)
        self.value = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
        self.adv   = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,num_actions))

    def _conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return F.relu(self.bn4(self.conv4(x)))

    def forward(self, x):
        B, T, R, _ = x.shape
        x = x.view(B*T, R, 2).permute(0,2,1)
        c = self._conv(x).flatten(1).view(B, T, -1)
        t = self.transformer(c.permute(1,0,2)).permute(1,0,2)
        ctx = self.ln(t).mean(1)
        v   = self.value(ctx)
        a   = self.adv(ctx)
        return v + a - a.mean(1,keepdim=True)


class DuelingTransformerNet(nn.Module):
    def __init__(self, obs_dim, num_actions, n_heads=4, n_layers=2, embed_dim=64):
        super().__init__()
        self.state_dim = 1
        self.lidar_dim = obs_dim - self.state_dim
        self.lidar_tokens    = 6 if self.lidar_dim>0 else 0
        self.points_per_token = max(1, self.lidar_dim//self.lidar_tokens) if self.lidar_tokens>0 else 0
        self.num_tokens      = self.lidar_tokens + (1 if self.state_dim>0 else 0)

        if self.state_dim>0:
            self.state_embedding = nn.Linear(self.state_dim, embed_dim)
        if self.lidar_tokens>0:
            self.lidar_embedding = nn.Linear(self.points_per_token, embed_dim)

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                         dim_feedforward=4*embed_dim,
                                         batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.adv_fc = nn.Sequential(nn.Linear(self.num_tokens*embed_dim,256), nn.ReLU())
        self.val_fc = nn.Sequential(nn.Linear(self.num_tokens*embed_dim,256), nn.ReLU())
        self.advantage_head = nn.Linear(256, num_actions)
        self.value_head     = nn.Linear(256, 1)

    def forward(self, obs):
        B = obs.size(0)
        tokens = []
        if self.state_dim>0:
            tokens.append(self.state_embedding(obs[:, :self.state_dim]).unsqueeze(1))
        if self.lidar_tokens>0:
            ld = obs[:, self.state_dim:]
            for i in range(self.lidar_tokens):
                start = i*self.points_per_token
                chunk = ld[:, start:start+self.points_per_token]
                if chunk.size(1)<self.points_per_token:
                    chunk = F.pad(chunk, (0,self.points_per_token-chunk.size(1)))
                tokens.append(self.lidar_embedding(chunk).unsqueeze(1))
        seq = torch.cat(tokens, dim=1) if tokens else torch.zeros(B,0)
        seq = seq + self.pos_embedding[:, :seq.size(1), :]
        out = self.transformer(seq)
        flat = out.reshape(B, -1)
        adv = self.adv_fc(flat)
        val = self.val_fc(flat)
        A = self.advantage_head(adv)
        V = self.value_head(val)
        return V + A - A.mean(1, keepdim=True)
