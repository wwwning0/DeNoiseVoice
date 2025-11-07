# model/dccrn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_re = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_im = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        real = self.conv_re(x.real) - self.conv_im(x.imag)
        imag = self.conv_re(x.imag) + self.conv_im(x.real)
        return torch.complex(real, imag)

class ComplexGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.gru_re = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru_im = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: (B, T, F)
        real, _ = self.gru_re(x.real)
        imag, _ = self.gru_im(x.imag)
        return torch.complex(real, imag)

class DCCRN(nn.Module):
    def __init__(self, n_fft=512, hop=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.window = torch.hann_window(n_fft)

        # Encoder
        self.enc1 = ComplexConv2d(2, 32, (7, 5), stride=(2, 2), padding=(3, 2))
        self.enc2 = ComplexConv2d(32, 64, (5, 3), stride=(2, 2), padding=(2, 1))
        self.enc3 = ComplexConv2d(64, 128, (5, 3), stride=(2, 2), padding=(2, 1))
        self.enc4 = ComplexConv2d(128, 256, (5, 3), stride=(1, 1), padding=(2, 1))

        # Bottleneck
        self.gru = ComplexGRU(256 * (n_fft//8 + 1), 256 * (n_fft//8 + 1))

        # Decoder
        self.dec4 = nn.ConvTranspose2d(512, 128, (5, 3), stride=(1, 1), padding=(2, 1))
        self.dec3 = nn.ConvTranspose2d(256, 64, (5, 3), stride=(2, 2), padding=(2, 1), output_padding=(1, 1))
        self.dec2 = nn.ConvTranspose2d(128, 32, (5, 3), stride=(2, 2), padding=(2, 1), output_padding=(1, 1))
        self.dec1 = nn.ConvTranspose2d(64, 2, (7, 5), stride=(2, 2), padding=(3, 2), output_padding=(1, 1))

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        # STFT → complex tensor (B, F, T)
        complex_spec = torch.stft(
            x, self.n_fft, self.hop,
            window=self.window.to(x.device),
            return_complex=True
        )  # (B, F, T)

        # 拆分为实部和虚部，并拼接为 (B, 2, F, T)
        spec = torch.stack([complex_spec.real, complex_spec.imag], dim=1)  # (B, 2, F, T)

        # Encoder: 输入为 (B, 2, F, T)，输出为实数张量
        e1 = F.relu(self.enc1(spec))      # (B, 32, F1, T1)
        e2 = F.relu(self.enc2(e1))        # (B, 64, F2, T2)
        e3 = F.relu(self.enc3(e2))        # (B, 128, F3, T3)
        e4 = F.relu(self.enc4(e3))        # (B, 256, F4, T4)

        # GRU bottleneck
        B, C, FF, T_enc = e4.shape
        gru_in = e4.permute(0, 3, 1, 2).reshape(B, T_enc, -1)  # (B, T_enc, C*F)
        gru_out = self.gru(gru_in)  # (B, T_enc, C*F), complex
        # 将复数 GRU 输出拆为实虚并拼接 → (B, 2*C, F, T_enc)
        gru_out_real = gru_out.real.reshape(B, T_enc, C, FF).permute(0, 2, 3, 1)  # (B, C, F, T_enc)
        gru_out_imag = gru_out.imag.reshape(B, T_enc, C, FF).permute(0, 2, 3, 1)  # (B, C, F, T_enc)
        gru_out_cat = torch.cat([gru_out_real, gru_out_imag], dim=1)  # (B, 2*C, F, T_enc)

        # Decoder with skip connections
        d4 = F.relu(self.dec4(gru_out_cat))  # (B, 128, F3, T3)
        d3 = F.relu(self.dec3(torch.cat([d4, e3], dim=1)))  # e3 is (B, 128, F3, T3)
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (B, 2, F, T)

        # 构建复数 mask
        mask = torch.complex(d1[:, 0], d1[:, 1])  # (B, F, T)
        # 应用 mask 到原始复数谱
        enhanced_spec = mask * complex_spec  # (B, F, T)
        # iSTFT 重建语音
        enhanced = torch.istft(
            enhanced_spec,
            self.n_fft,
            self.hop,
            window=self.window.to(x.device),
            length=T
        )
        return enhanced  # (B, T)