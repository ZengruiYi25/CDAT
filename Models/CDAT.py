import torch
from torch import nn
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class DepthWiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.BatchNorm1d(dim_in),
            nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ConvDualAttention(nn.Module):
    def __init__(self, dim, proj_kernel, heads=8, dropout=0.):
        super().__init__()
        inner_dim = dim * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = DepthWiseConv1d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_k = DepthWiseConv1d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_v = DepthWiseConv1d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.DWConv = DepthWiseConv1d(dim, dim, kernel_size=1, padding=0, stride=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x -> (b h) (x) d', h=self.heads), (q, k, v))

        # =============================
        k = torch.bmm(q.permute(0, 2, 1), self.softmax(k)) * self.scale
        q = self.sigmoid(self.DWConv(q.permute(0, 2, 1)).permute(0, 2, 1)) * v

        out = torch.bmm(v, k) + q

        out = rearrange(out, '(b h) (x) d -> b (h d) x', h=self.heads)

        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, heads, mlp_mult=4, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, ConvDualAttention(dim, proj_kernel=proj_kernel, heads=heads, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class CDAT(nn.Module):
    def __init__(self, num_classes, emb_dim, emb_kernel, emb_stride, proj_kernel, heads, mlp_mult, dropout=0.):
        super().__init__()

        layers = []
        for stage in range(4):
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels=2 if stage == 0 else emb_dim[stage-1],
                          out_channels=emb_dim[stage],
                          kernel_size=emb_kernel[stage],
                          stride=emb_stride[stage],
                          padding=emb_stride[stage] // 2),
                LayerNorm(emb_dim[stage]),
                Transformer(dim=emb_dim[stage], proj_kernel=proj_kernel[stage], heads=heads[stage],
                            mlp_mult=mlp_mult[stage], dropout=dropout)
            ))

        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(emb_dim[-1], num_classes)


    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x).squeeze(-1)
        x = self.out(x)
        return x

if __name__ == '__main__':
    from thop import profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    num_classes = 11
    emb_dim = [32, 32, 32, 32]
    emb_kernel = [7, 5, 3, 3]
    emb_stride = [2, 2, 2, 2]
    proj_kernel = [3, 3, 3, 3]
    heads = [4, 4, 4, 4]
    mlp_mult = [4, 4, 4, 4]
    dropout = 0.

    net = CDAT(num_classes=num_classes,
               emb_dim=emb_dim,
               emb_kernel=emb_kernel,
               emb_stride=emb_stride,
               proj_kernel=proj_kernel,
               heads=heads,
               mlp_mult=mlp_mult,
               dropout=dropout).to(device)

    # Test
    print(" ================== Evaluating ==================  ")
    net.eval()
    flops, params = profile(net, inputs=(torch.randn((1, 2, 128), device=device),), verbose=False)
    txt0 = "The number of parameters is %d(%.2fM)(%.2fK)\nThe number of flops is %d(%.2fM)" % (
        params, params / 1e6, params / 1e3, flops, flops / 1e6)
    print(txt0)
    print(" ================================================  ")