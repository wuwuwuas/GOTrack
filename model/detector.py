import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0'


def loss_metric(y_pred, y_true):
    y_pred = torch.round(y_pred)
    y_true = torch.round(y_true)

    num_p_gt = torch.sum(y_true == 1.0, dim=[1, 2, 3])
    num_p_pre = torch.sum(y_pred == 1.0, dim=[1, 2, 3])

    miss_num_p_pre = torch.zeros(y_true.size(0), device=y_true.device)  # [B]
    false_num_p_pre = torch.zeros(y_true.size(0), device=y_true.device)  # [B]
    for i in range(y_true.size(0)):
        p_position_gt = (y_true[i] == 1.0)
        p_position_pre = (y_pred[i] == 1.0)

        miss_num_p_pre[i] = torch.sum((y_pred[i][p_position_gt] != 1.0).float())
        false_num_p_pre[i] = torch.sum((y_true[i][p_position_pre] != 1.0).float())

    miss_rate = miss_num_p_pre / (num_p_gt)
    acc_rate = 1 - miss_rate
    yield_rate = 1 - (false_num_p_pre / (num_p_pre + 1e-8))

    avg_miss_rate = torch.mean(miss_rate).item()
    avg_acc_rate = torch.mean(acc_rate).item()
    avg_yield_rate = torch.mean(yield_rate).item()

    metrics = {
        'miss_rate': avg_miss_rate,
        'acc_rate': avg_acc_rate,
        'yield_rate': avg_yield_rate,
    }

    return metrics
def RMSE_loss(y_pred, y_true):
    sq = torch.square(y_true-y_pred)
    sum_sq = torch.sum(torch.sum(sq.squeeze(1), dim = 1), dim = 1)
    mean_sum = sum_sq / 256/256
    rmse_loss = torch.mean(torch.sqrt(mean_sum))

    return rmse_loss


class CBAMLayer(nn.Module):
    def __init__(self, channel):
        super(CBAMLayer, self).__init__()
 
        # channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp1 = nn.Sequential(
            nn.Conv2d(channel, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channel, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(channel, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channel, 1)
        )
        
        # spatial attention
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channel, 1)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp1(self.max_pool(x))
        avg_out = self.mlp2(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 8,7)
        self.encoder2 = self.conv_block(8, 16,3)
        self.encoder3 = self.conv_block(16, 32, 3)
        self.encoder4 = self.conv_block(32, 32, 3)
        
        # Bottleneck
        self.bottleneck = self.conv_block(32, 32)
        
        # context
        self.context1 = self.conv_block(in_channels, 8,7)
        self.context2 = self.conv_block(8, 16,3)
        self.context3 = self.conv_block(16, 32,3)
        self.context4 = self.conv_block(32, 16,3)
        self.context5 = nn.Conv2d(16, out_channels, kernel_size=1)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(64, 32)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(32, 16)
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(16, 8)
        
        self.conv1 = self.conv_block(8, 16)
        self.weight = CBAMLayer(2)

        # Output
        self.conv_last = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def conv_block(self, in_channels, out_channels,kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        # Encoder
        e1 = self.encoder1(x)
        e1_d = F.max_pool2d(e1, kernel_size=2)
        
        e2 = self.encoder2(e1_d)
        e2_d = F.max_pool2d(e2, kernel_size=2)
        
        e3 = self.encoder3(e2_d)
        e3_d = F.max_pool2d(e3, kernel_size=2)

        e4 = self.encoder4(e3_d)
        # Bottleneck
        b = self.bottleneck(e4)
        
        # context
        con1 = self.context1(x)
        con1 = self.context2(con1)
        con1 = self.context3(con1)
        con1 = self.context4(con1)
        con1 = self.context5(con1)
        # b = self.trans(b)
        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        
        
        c1 = self.conv1(d1)
        # Output
        out = self.conv_last(c1)
        out = self.weight(torch.cat((out, con1),dim=1))
        out = torch.sum(out, dim=1, keepdim=True)
        out = torch.clamp(out, min=0.0, max=1.0)
        
        return out

