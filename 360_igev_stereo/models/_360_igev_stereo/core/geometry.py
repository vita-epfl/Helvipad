import torch
import torch.nn.functional as F


def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    xgrid, ygrid = coords.split([1,1], dim=-1)
    ygrid = 2*ygrid/(H-1) - 1
    assert torch.unique(xgrid).numel() == 1 and W == 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, h2, _ = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, d, 1)

        # init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        init_corr = init_corr.reshape(b*h*w, 1, h2, 1)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            # geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            geo_volume = F.avg_pool2d(geo_volume, [2,1], stride=[2,1])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            # init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            init_corr = F.avg_pool2d(init_corr, [2,1], stride=[2,1])
            self.init_corr_pyramid.append(init_corr)




    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dy = torch.linspace(-r, r, 2*r+1)
            dy = dy.view(1, 2*r+1, 1, 1).to(disp.device)
            y0 = dy + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            x0 = torch.zeros_like(y0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_y0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dy
            init_coords_lvl = torch.cat([x0, init_y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H1, W = fmap1.shape
        _, _, H2, _ = fmap2.shape
        fmap1 = fmap1.view(B, D, H1, W)
        fmap2 = fmap2.view(B, D, H2, W)
        corr = torch.einsum('aijk,aihk->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H1, W, H2, 1).contiguous()
        return corr