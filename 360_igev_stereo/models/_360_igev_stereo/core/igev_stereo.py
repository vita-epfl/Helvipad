import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder, Feature, get_polar_map
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../general')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from conversion import disp_deg_to_disp_pix


autocast = torch.amp.autocast

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class IGEVStereo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = self.cfg.dataset
        self.model_name = self.cfg.model_name

        self.min_disp_pix = disp_deg_to_disp_pix(self.cfg.min_disp_deg, dataset=self.dataset)
        self.max_disp_pix = disp_deg_to_disp_pix(self.cfg.max_disp_deg, dataset=self.dataset)
        
        context_dims = self.cfg.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[self.cfg.hidden_dims, context_dims], norm_fn="batch", downsample=cfg.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.cfg, hidden_dims=self.cfg.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], self.cfg.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.cfg.n_gru_layers)])

        self.feature = Feature()

        self.coord2_net = nn.Sequential(BasicConv(1, 32, kernel_size=3, stride=2, padding=1))
        self.coord4_net = nn.Sequential(BasicConv(32, 32, kernel_size=3, stride=2, padding=1))
        self.coord32_net = nn.Sequential(BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
                                        BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
                                        BasicConv(32, 32, kernel_size=3, stride=2, padding=1))
        
        self.stem2_pre = nn.Sequential(
                BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1)
                )
        self.stem_2_post = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )

        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast('cuda', enabled=self.cfg.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def forward(self, image_bottom, image_top, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image_bottom = (2 * (image_bottom / 255.0) - 1.0).contiguous()
        image_top = (2 * (image_top / 255.0) - 1.0).contiguous()
        height, width = image_bottom.size(2), image_bottom.size(3)

        polar_map = get_polar_map(image_bottom.size(0), height, width, device=image_bottom.device, dataset=self.dataset)
        polar_map = (2 * (polar_map / 180.0) - 1.0).contiguous()
        polar2 = self.coord2_net(polar_map)
        polar4 = self.coord4_net(polar2)
        polar32 = self.coord32_net(polar4)

        with autocast('cuda', enabled=self.cfg.mixed_precision):
            features_bottom = self.feature(image_bottom, polar32)
            features_top = self.feature(image_top, polar32)

            stem_2b = self.stem2_pre(image_bottom)
            stem_2b_concat = torch.cat((stem_2b, polar2), 1)
            stem_2b = self.stem_2_post(stem_2b_concat)
            stem_2t = self.stem2_pre(image_top)
            stem_2t_concat = torch.cat((stem_2t, polar2), 1)
            stem_2t = self.stem_2_post(stem_2t_concat)
            stem_4b = self.stem_4(stem_2b)
            stem_4t = self.stem_4(stem_2t)
            features_bottom[0] = torch.cat((features_bottom[0], stem_4b), 1)
            features_top[0] = torch.cat((features_top[0], stem_4t), 1)

            match_bottom = self.desc(self.conv(features_bottom[0]))
            match_top = self.desc(self.conv(features_top[0]))
            gwc_volume = build_gwc_volume(match_bottom, match_top, self.cfg.max_disp//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_bottom[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_bottom)

            # Init disp from geometry encoding volume --> this corresponds to part 3.3 
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.cfg.max_disp//4)
            init_disp = torch.clamp(init_disp, self.min_disp_pix/4, self.max_disp_pix/4)
            
            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(features_bottom[0])
                xspx = self.spx_2(xspx, stem_2b)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(image_bottom, num_layers=self.cfg.n_gru_layers, polar4=polar4)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]


        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_bottom.float(), match_top.float(), geo_encoding_volume.float(), radius=self.cfg.corr_radius, num_levels=self.cfg.corr_levels)
        b, c, h, w = match_bottom.shape
        coords = torch.arange(h).float().to(match_bottom.device).reshape(1,h,1,1).repeat(b, 1, w, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast('cuda', enabled=self.cfg.mixed_precision):
                if self.cfg.n_gru_layers == 3 and self.cfg.slow_fast_gru: # Update low-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.cfg.n_gru_layers >= 2 and self.cfg.slow_fast_gru:# Update low-res ConvGRU and mid-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=self.cfg.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.cfg.n_gru_layers==3, iter08=self.cfg.n_gru_layers>=2)

            disp = disp + delta_disp
            disp = torch.clamp(disp, self.min_disp_pix/4, self.max_disp_pix/4)
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2b)
            disp_up = torch.clamp(disp_up, self.min_disp_pix, self.max_disp_pix)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        init_disp = torch.clamp(init_disp, self.min_disp_pix, self.max_disp_pix)
        return init_disp, disp_preds
