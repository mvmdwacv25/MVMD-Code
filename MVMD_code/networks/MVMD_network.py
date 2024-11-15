import torch.nn as nn
import torch
import torch.nn.functional as F
from .DeepLabV3 import DeepLabV3
from .CBAM import ChannelGate, CBAM

class MVMD_Network(nn.Module):
    def __init__(self, pretrained_path=None):
        super(MVMD_Network, self).__init__()

        self.imgconv = DeepLabV3()

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            print(f"Load checkpoint:{pretrained_path}")
            self.encoder.load_state_dict(checkpoint['model'])

        ################################## img block ######################################
        # self.imgconv = ()
        self.self_attention_img = Self_Relation_Attention(in_channels=256, out_channels=256)
        self.self_attention_img_low = Self_Relation_Attention(in_channels=256, out_channels=256)
        self.ra_attention_cross = Relation_Attention(in_channels=256, out_channels=256)
        self.ra_attention_cross_low = Relation_Attention(in_channels=256, out_channels=256)

        self.img_low_project= nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.imginfoconv = nn.Sequential(
            nn.Conv2d(512, 384, kernel_size=5, padding=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.cbam_img = CBAM(gate_channels=256, no_spatial=True)

        ############################# in & out connection #################################
        self.connect_crossattn = Relation_Attention(in_channels=256, out_channels=256)

        self.connect_conv = nn.Sequential(
            nn.Conv2d(256, 256, stride=2, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.cbam_connect = CBAM(gate_channels=256, no_spatial=True)


        ################################ inverse block ########################################
        self.lowfeatconv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        
        
        self.infoconv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            )

        self.coarsemaskconv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )
        

        self.mask_fine_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),   
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )

        self.high_level_conv = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # ####################################### minus net ###########################################
        # identify the discontinuity of mirror by img
        self.local_feature = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.context_feature = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.img_bn_relu = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


    def forward(self, img1, img2, img3):

        h = img1.size(-2)
        w = img1.size(-1)

        ################################# img relation block #####################################
        imginfo1, imginfo1_low = self.imgconv(img1)
        imginfo2, imginfo2_low = self.imgconv(img2)
        imginfo3, imginfo3_low = self.imgconv(img3)
        
        h_low = imginfo1_low.size(-2)
        w_low = imginfo1_low.size(-1)

        imgcross1a, _ = self.ra_attention_cross(imginfo1, imginfo2)
        imgself1a = self.self_attention_img(imgcross1a)

        imgcross1b, _ = self.ra_attention_cross(imginfo1, imginfo3)
        imgself1b = self.self_attention_img(imgcross1b)

        imgself = torch.cat((imgself1a, imgself1b), dim=1)

        imginfo_o = self.imginfoconv(imgself)
        imginfo = self.cbam_img(imginfo_o)

        ################################ in & out connection block ########################################
        imginfo_reshaped =  F.interpolate(imginfo_o, size=imginfo1_low.shape[2:], mode='bilinear', align_corners=False)
        imginfo_reshaped = F.normalize(imginfo_reshaped, p=2, dim=1)

        imginfo1_low_flipped = torch.flip(imginfo1_low, dims=[-1])

        imginfo1_mirror = imginfo1_low * imginfo_reshaped
        imginfo1_env = imginfo1_low_flipped * (1. - imginfo_reshaped)

        img_connect, _ = self.connect_crossattn(imginfo1_mirror, imginfo1_env)

        connectinfo = self.connect_conv(img_connect)
        connectinfo = self.cbam_connect(connectinfo)

        ##################################### reverse block #################################
        imginfo1_low_feat = self.lowfeatconv(imginfo1_low)
        info = self.infoconv(connectinfo + imginfo)

        mask_coarse = self.coarsemaskconv(info)# (1, 1, 60, 80)
        mask_c = F.interpolate(mask_coarse, (h,w), mode='bilinear')  # (1, 1, 480, 640)

        info_project = F.interpolate(info, (h_low,w_low), mode='bilinear')
        mask_final_info = self.mask_fine_conv(info_project)

        # edge extract net
        imginfo1_low = self.high_level_conv(imginfo1_low_feat)

        local_img = self.local_feature(imginfo1_low + mask_final_info)
        context_img = self.context_feature(imginfo1_low + mask_final_info)
        contrast_img = self.img_bn_relu(local_img - context_img)  # out (1, 256, 120, 160)

        mask_fine = self.final_conv(contrast_img)
        mask_f = F.interpolate(mask_fine, (h,w), mode='bilinear')

        return mask_c, mask_f


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class RAttention(nn.Module):
    '''This part of code is refactored based on https://github.com/Serge-weihao/CCNet-Pure-Pytorch.
       We would like to thank Serge-weihao and the authors of CCNet for their clear implementation.'''

    def __init__(self, in_dim):
        super(RAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF

    def forward(self, x_exmplar, x_query):
        m_batchsize, _, height, width = x_query.size()
        proj_query = self.query_conv(x_query)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_query_LR = torch.diagonal(proj_query, 0, 2, 3)
        proj_query_RL = torch.diagonal(torch.transpose(proj_query, 2, 3), 0, 2, 3)

        proj_key = self.key_conv(x_exmplar)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_key_LR = torch.diagonal(proj_key, 0, 2, 3).permute(0, 2, 1).contiguous()
        proj_key_RL = torch.diagonal(torch.transpose(proj_key, 2, 3), 0, 2, 3).permute(0, 2, 1).contiguous()

        proj_value = self.value_conv(x_exmplar)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value_LR = torch.diagonal(proj_value, 0, 2, 3)
        proj_value_RL = torch.diagonal(torch.transpose(proj_value, 2, 3), 0, 2, 3)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        energy_LR = torch.bmm(proj_key_LR, proj_query_LR)
        energy_RL = torch.bmm(proj_key_RL, proj_query_RL)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        out_LR = self.softmax(torch.bmm(proj_value_LR, energy_LR).unsqueeze(-1))
        out_RL = self.softmax(torch.bmm(proj_value_RL, energy_RL).unsqueeze(-1))

        out = out_H + out_W + out_LR + out_RL + x_exmplar
        return out


class Relation_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Relation_Attention, self).__init__()
        inter_channels = in_channels // 4
        self.conv_examplar = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.conv_query = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.ra = RAttention(inter_channels)
        self.conv_examplar_tail = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),nn.ReLU(inplace=False))
        self.conv_query_tail = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),nn.ReLU(inplace=False))

            
    def forward(self, x_exmplar, x_query, recurrence=2):

        x_exmplar = self.conv_examplar(x_exmplar)
        x_query = self.conv_query(x_query)
        for i in range(recurrence):
            x_exmplar = self.ra(x_exmplar, x_query)
            x_query = self.ra(x_query, x_exmplar)
            # x_exmplar, x_query = self.ra(x_exmplar, x_query)
        x_exmplar_out = self.conv_examplar_tail(x_exmplar)
        x_query_out = self.conv_query_tail(x_query)
        return x_exmplar_out, x_query_out


class Self_Relation_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Self_Relation_Attention, self).__init__()
        inter_channels = in_channels // 4
        self.conv_examplar = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                           nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=False))
        self.conv_query = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=False))

        self.ra = RAttention(inter_channels)
        self.conv_examplar_tail = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))
        self.conv_query_tail = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                             nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x, recurrence=2):

        x_exmplar = self.conv_examplar(x)
        x_query = self.conv_query(x)
        for i in range(recurrence):
            x_exmplar = self.ra(x_exmplar, x_query)
        x_exmplar_out = self.conv_examplar_tail(x_exmplar)
        return x_exmplar_out

class CoattentionModel(nn.Module):  # spatial and channel attention module
    def __init__(self, num_classes=1, all_channel=256, all_dim=26 * 26):  # 473./8=60 416./8=52
        super(CoattentionModel, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate1 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate2 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.globalAvgPool = nn.AvgPool2d(26, stride=1)
        self.fc1 = nn.Linear(in_features=256*2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=256)
        self.fc3 = nn.Linear(in_features=256*2, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, exemplar, query):

        # spatial co-attention
        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        A1 = F.softmax(A.clone(), dim=1)  #
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        
        # spacial attention
        input1_mask = self.gate1(torch.cat([input1_att, input2_att], dim=1))
        input2_mask = self.gate2(torch.cat([input1_att, input2_att], dim=1))
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)

        # channel attention
        out_e = self.globalAvgPool(torch.cat([input1_att, input2_att], dim=1))
        out_e = out_e.view(out_e.size(0), -1)
        out_e = self.fc1(out_e)
        out_e = self.relu(out_e)
        out_e = self.fc2(out_e)
        out_e = self.sigmoid(out_e)
        out_e = out_e.view(out_e.size(0), out_e.size(1), 1, 1)
        out_q = self.globalAvgPool(torch.cat([input1_att, input2_att], dim=1))
        out_q = out_q.view(out_q.size(0), -1)
        out_q = self.fc3(out_q)
        out_q = self.relu(out_q)
        out_q = self.fc4(out_q)
        out_q = self.sigmoid(out_q)
        out_q = out_q.view(out_q.size(0), out_q.size(1), 1, 1)

        # apply dual attention masks
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input2_att = out_e * input2_att
        input1_att = out_q * input1_att

        # concate original feature
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        input1_att = self.bn1(input1_att)
        input2_att = self.bn2(input2_att)
        input1_att = self.prelu(input1_att)
        input2_att = self.prelu(input2_att)

        return input1_att, input2_att  # shape: NxCx

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

