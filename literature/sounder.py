import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat
import json

def double_conv(in_c, out_c):
    conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.005, inplace=True),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.005, inplace=True),
            nn.BatchNorm2d(out_c)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_height = target_tensor.size()[2]
    tensor_height = tensor.size()[2]
    delta = tensor_height - target_height
    delta = delta // 2

    target_width = target_tensor.size()[3]
    tensor_width = tensor.size()[3]

    gama = tensor_width - target_width
    gama = gama // 2
    # print(target_size, tensor_size, delta)
    return tensor[:, :, delta:tensor_height - delta, gama:tensor_width - gama]

def equalize_tensor(t1, t2):

    t1_h, t1_w = t1.size()[2], t1.size()[3]
    t2_h, t2_w = t2.size()[2], t2.size()[3]

    min_h, min_w = min(t1_h, t2_h), min(t1_w, t2_w)

    t1 = t1[:, :, :min_h, :min_w]
    t2 = t2[:, :, :min_h, :min_w]

    #print(t1.size(), t2.size(), 'this is new')

    return t1, t2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

# defining Hyperparameters

class PatchEmbedding(nn.Module):
    """Split the image into patches and then embed them.

    Parameters
    __________
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
#        print(self.n_patches, 'the number of patches')

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,)

    def forward(self, x):

        """Run forward pass"""

        x = self.proj(x)
#        print(x.size())
        x = x.flatten(2)
#        print(x.size())
        x = x.transpose(1, 2)
#        print(x.size())

        return x


class Attention(nn.Module):

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.attn_p = attn_p
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass. """

        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)
#        print(qkv.size())
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
#        print(qkv.size())
        qkv = qkv.permute(2, 0, 3, 1, 4)
#        print(qkv.size())
        q, k, v = qkv[0], qkv[1], qkv[2]
#        print(q.size(), k.size(), v.size())
        k_t = k.transpose(-2, -1)
#        print(k_t.size())

        dp = (
            q @ k_t
        ) * self.scale

#        print(dp.size(), 'the size of the dot product')
        attn = dp.softmax(dim=1)
#        print(attn.size(), 'the size of the attention module')
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
#        print(weighted_avg.size())
        weighted_avg = weighted_avg.transpose(
            1, 2
        )
#        print(weighted_avg.size())

        weighted_avg = weighted_avg.flatten(2)
#        print(weighted_avg.size())
        x = self.proj(weighted_avg)
#        print(x.size())
        x = self.proj_drop(x)
#        print(x.size())

        return x


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
#        x = self.drop(x)
        x = self.fc2(x)

        return x


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim,
                              n_heads=n_heads,
                              qkv_bias=qkv_bias,
                              attn_p=attn_p,
                              proj_p=p
                              )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=hidden_features,
                       out_features=dim,
                       )

    def forward(self, x):
        """Run Forward Pass"""

        x = x + self.attn(self.norm1(x))
#        print(x.size(), 'the size of the tensor in Block module')
        x = x + self.mlp(self.norm2(x))
#        print(x.size(), 'the size of the tensor in Block module')

        return x


class VisionTransformer(nn.Module):
    """
    Simplified implementation of the Vision transformer
    Parameters
    ---------
    """
    def __init__(self,
                 img_size=25,
                 patch_size=1,
                 in_chans=1024,
                 n_classes=1000,
                 embed_dim=768,
                 depth=8,
                 n_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 p=0.,
                 attn_p=0.,
                 ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim,
                                      )
    #    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList([Block(dim=embed_dim,
                                           n_heads=n_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           p=p,
                                           attn_p=attn_p,
                                           )
                                    for _ in range(depth)
                                     ]
                                    )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
#        print(x.size(), 'the size of the patch embedding')

    #    cls_token = self.cls_token.expand(n_samples, -1, -1)
    #    print(cls_token.size(), 'the size of the class token')
    #    x = torch.cat((cls_token, x), dim=1)
#        print(x.size(), self.pos_embed.size(), 'the size of the position embedding')
        x = x + self.pos_embed
#        print(x.size(), 'the size of the position embedding')
        x = self.pos_drop(x)
#        print(x.size(), 'the size of the concatenated embedding')

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x_size = x.size()

        x = torch.reshape(x, (x_size[0],
                              int(x_size[1] ** 0.5),
                              int(x_size[1] ** 0.5),
                              x_size[2]
                              )
                          )
#        print(x.size())

        x = torch.reshape(x, (x_size[0],
                              x_size[2],
                              int(x_size[1] ** 0.5),
                              int(x_size[1] ** 0.5)
                              )
                          )
#        print(x.size())

    #    cls_token_final = x[:, 0]
    #    x = self.head(cls_token_final)
    #    print(x.size(), 'the dimension of the class token tensor')

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool=['max', 'avg']):

        super(ChannelAttention, self).__init__()

        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
                )

        self.pool = pool

    def forward(self, x):
        channel_attn_sum = None

        for pool in self.pool:
            if pool == 'max':
                max_pool = F.max_pool2d(x,
                        (x.size(2), x.size(3)),
                        stride=(x.size(2), x.size(3)))

                channel_attn_raw = self.mlp( max_pool )

            elif pool == 'avg':
                avg_pool = F.avg_pool2d(x,
                        (x.size(2), x.size(3)),
                        stride=(x.size(2), x.size(3)))

                channel_attn_raw = self.mlp( avg_pool )

            if channel_attn_sum is None:
                channel_attn_sum = channel_attn_raw

            else:
                channel_attn_sum += channel_attn_raw

#        print('Details of Channel Attention Gate')

#        print(x.size(), 'The size of the input')

        scale = torch.sigmoid( channel_attn_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)

#        print(scale.size())

#        print((x*scale).size())

        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
            torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttention(nn.Module):

    def __init__(self):

        super(SpatialAttention, self).__init__()

        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
                2,
                1,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                relu=False)

    def forward(self, x):
#        print(x.size(), 'the size of the input')

        x_compress = self.compress(x)

#       print(x_compress.size(), 'output after channel pooling')

        x_out = self.spatial(x_compress)

#       print(x_out.size(), 'output after Convolution' )

        scale = torch.sigmoid(x_out) #broadcasting

#       print(scale.size(), 'output after sigmoid')

#       print((x*scale).size(), 'output after element wise multiplication')

        return x * scale


class ResBlock(nn.Module):

    def __init__(self, in_c):
        super(ResBlock, self).__init__()
        self.in_c = in_c

        self.conv = double_conv(in_c, in_c)

    def forward(self, x):

        """ Run forward pass """

        residual = x
        out = self.conv(x)
        out += residual

        #out = out.view(out.size(0), -1)
#        print(out.size())

        return out


class BasicConv(nn.Module):

    def __init__(self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
            bias=False):

        super(BasicConv, self).__init__()

        self.out_channels = out_planes
        self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias)
        self.bn = nn.BatchNorm2d(
                out_planes,
                eps=1e-5,
                momentum=0.01,
                affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


class BiFusion(nn.Module):
    def __init__(self, tf_channel, cnn_channel):
        super(BiFusion, self).__init__()

        if tf_channel != cnn_channel:
            print('The dimension is wrong')

#        print("Entering the BiFusuion Module")

        self.tf_channel = tf_channel
        self.cnn_channel = cnn_channel

        self.spatial_attn = SpatialAttention()

        self.channel_attn = ChannelAttention(tf_channel)

        self.conv = nn.Sequential(
            nn.Conv2d(tf_channel, tf_channel // 2, kernel_size=(3,3), stride=1, padding=1),
            nn.LeakyReLU(0.005, inplace=True),
            nn.BatchNorm2d(tf_channel // 2)
        )

        self.res_block =  ResBlock(int(2.5 * tf_channel))

    def forward(self, tf, cnn):

        tf = self.channel_attn(tf)
        cnn = self.spatial_attn(cnn)

        tf_prod_cnn = tf * cnn

        tf_prod_cnn_conv = self.conv(tf_prod_cnn)

        concat = torch.cat([tf,
                            tf_prod_cnn_conv,
                            cnn], dim=1)

        out = self.res_block(concat)

        return out


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

    def forward(self, image):
        x1 = self.down_conv_1(image)
#        print(x1.size())
        x2 = self.max_pool_2x2(x1)
#        print(x2.size(), 'the size of the x2')
        x3 = self.down_conv_2(x2)
#        print(x3.size())
        x4 = self.max_pool_2x2(x3)
#        print(x4.size(), 'the size of the x4')
        x5 = self.down_conv_3(x4)
#        print(x5.size())
        x6 = self.max_pool_2x2(x5)
#        print(x6.size(), 'the size of the x6')
        x7 = self.down_conv_4(x6)
#        print(x7.size())
        x8 = self.max_pool_2x2(x7)
#        print(x8.size(), 'the size of the x8')
        x9 = self.down_conv_5(x8)
#        print(x9.size(), 'the size of the x9')

        return [x2, x4, x6, x8, x9]


class Decoder(nn.Module):

    def __init__(self, alpha, beta, gama,num_classes=5 ):

        super(Decoder, self).__init__()

        self.bf1 = BiFusion(512, 512)
        self.bf2 = BiFusion(256, 256)
        self.bf3 = BiFusion(128, 128)
        self.bf4 = BiFusion(64, 64)

        self.cnn = CNN()
        self.vtf = VisionTransformer()

        self.conv1 = double_conv(1920, 256)
        self.conv2 = double_conv(576, 128)
        self.conv3 = double_conv(288, 64)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=768,
                                             out_channels=256,
                                             kernel_size=2,
                                             stride=2
                                             )

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=2,
                                             stride=2
                                             )

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2
                                             )

        self.out1 = nn.Conv2d(in_channels=64,
                             out_channels=num_classes,
                             kernel_size=1)

        self.out2 = nn.Conv2d(in_channels=768,
                              out_channels=num_classes,
                              kernel_size=1)

        self.out3 = nn.Conv2d(in_channels=1280,
                              out_channels=num_classes,
                              kernel_size=1)

        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.num_classes = num_classes

    def forward(self, x):       
        h, w = x.shape[2], x.shape[3]
        x= F.interpolate(x, size=(400, 400), mode='bilinear', align_corners=False)


        lst_cnn = self.cnn(x)

        x2_cnn, x4_cnn, x6_cnn = lst_cnn[0], lst_cnn[1], lst_cnn[2]
        x8_cnn, x9_cnn = lst_cnn[3], lst_cnn[4]

        x9_vtf = self.vtf(x9_cnn)

#        print(x9_vtf.size(), "The size of the x9_vtf")
#        print(x6_cnn.size(), "The size of te x6_cnn")

        x6_tf_upsample = self.up_trans_1(x9_vtf)

        #x6_tf_upsample = F.interpolate(x9_vtf,
        #                               size=x6_cnn.size()[2:],
        #                               mode='bilinear',
        #                               align_corners=True)

#        print(x6_tf_upsample.size(), 'the size of the x6_tf')

        x4_tf_upsample = self.up_trans_2(x6_tf_upsample)

        #x4_tf_upsample = F.interpolate(x6_tf_upsample,
        #                               size=x4_cnn.size()[2:],
        #                               mode='bilinear',
        #                               align_corners=True)

        x2_tf_upsample = self.up_trans_3(x4_tf_upsample)


        #x2_tf_upsample = F.interpolate(x4_tf_upsample,
        #                               size=x2_cnn.size()[2:],
        #                               mode='bilinear',
        #                               align_corners=True)

        tf_head = F.interpolate(x9_vtf,
                                size=x.size()[2:],
                                mode='bilinear',
                                align_corners=True)

        tf_head = self.out2(tf_head)

#        print(tf_head.size(), 'the head of the transformer')

#        print(x9_vtf.size(), 'the size of the Transformer')

        x9_vtf = x9_vtf[:, :512, :, :]
        x9_cnn = x9_cnn[:, :512, :, :]

        bf1 = self.bf1(x9_vtf, x9_cnn)

        bf1_upsample = F.interpolate(bf1,
                                     size=x6_cnn.size()[2:],
                                     mode='bilinear',
                                     align_corners=True)

        bf_head = F.interpolate(bf1,
                                size=x.size()[2:],
                                mode='bilinear',
                                align_corners=True)

        bf_head = self.out3(bf_head)

#        print(bf_head.size(), 'the size of the bf head')

#        bf_head = self.out(bf_head)

#        print(bf1_upsample.size(), 'the size of bf1 upsampled features')

        x6_tf_upsample = x6_tf_upsample[:, :x6_cnn.size()[1], :, :]
        bf2 = self.bf2(x6_tf_upsample, x6_cnn)
        bf1_ups_cat_bf2 = torch.cat([bf1_upsample, bf2], dim=1)
        bf1_bf2_conv = self.conv1(bf1_ups_cat_bf2)
        bf1_bf2_conv_ups = F.interpolate(bf1_bf2_conv,
                                         size=x4_cnn.size()[2:],
                                         mode='bilinear',
                                         align_corners=True)

        x4_tf_upsample = x4_tf_upsample[:, :x4_cnn.size()[1], :, :]
        bf3 = self.bf3(x4_tf_upsample, x4_cnn)
        cat1 =torch.cat([bf1_bf2_conv_ups, bf3], dim=1)
        bf2_bf3_conv = self.conv2(cat1)
        bf2_bf3_conv_ups = F.interpolate(bf2_bf3_conv,
                                         size=x2_cnn.size()[2:],
                                         mode='bilinear',
                                         align_corners=True)

        x2_tf_upsample = x2_tf_upsample[:, :x2_cnn.size()[1], :, :]
        bf4 = self.bf4(x2_tf_upsample, x2_cnn)
        cat2 = torch.cat([bf2_bf3_conv_ups, bf4], dim=1)
        bf3_bf4_conv = self.conv3(cat2)
        bf3_bf4_conv_ups = F.interpolate(bf3_bf4_conv,
                                         size=x.size()[2:],
                                         mode='bilinear',
                                         align_corners=True)

#        print(bf3_bf4_conv_ups.size(), 'the size of the upconvolve bf3_bf4_conv_ups')

        ag_head = self.out1(bf3_bf4_conv_ups)


        total_head = self.alpha * ag_head + self.beta * bf_head + self.gama * tf_head
        total_head= F.interpolate(total_head, size=(h, w ), mode='bilinear', align_corners=False)
#        print(head.size(), 'the size of the head')

        return total_head










#        print(bf1_bf2_conv.size(), 'the size of the bf1_bf2_conv')

#        print(bf1_ups_cat_bf2.size(), 'the concatenated size of joint tensor')

#        print(bf2.size(), 'the size of the bf2 upsampled features')

#        print(bf1.size(), 'The Size of the BiFusion Model 1')
#        print(x8_cnn.size(), 'the size of the x8_cnn module')
#        print(x6_cnn.size(), 'the size of the x6_cnn Module')



################################## Test Cases #######################################

#if __name__ == "__main__":
#    image = torch.rand((16, 1, 400 , 400))
#    model = Decoder()

#    if torch.cuda.device_count() > 1:
#        print("let us use", torch.cuda.device_count(), "GPUs!")
#        model = nn.DataParallel(model)
#    model = model.to(device)
#    model = model.float()
#    x = model(image)

############################ Sample Experiments on the above ########################



#if __name__ == "__main__":

#    image = torch.rand((30, 1, 400, 400))
#    data = CNN()
#    x = data(image)
#    x = x[4]
#    print(x.size())

#    model = VisionTransformer()
    #x = model(x)

#    if torch.cuda.device_count() > 1:
#        print("let us use", torch.cuda.device_count(), "GPUs!")
#        model = nn.DataParallel(model)

#    model = model.to(device)
#    model = model.float()
    #image = torch.rand((30, 3, 512, 512))
    #model = VisionTransformer()
#    x = model(x)

#***************************** CNN EXPERIMENTATION ****************************#

#if __name__ == "__main__":
#    image = torch.rand((16, 1, 400, 400))
#    model = Decoder()
#    x = model(image)
    #print(x[4].size())
#    model1 = VisionTransformer()
#    y = model1(x[4])
#    print(y.size())


#***************************** BIFUSION MODULE EXPERIMENTATION **************#

#if __name__ == "__main__":
#    image1 = torch.rand(16, 256, 20, 20)
#    image2 = torch.rand(16, 256, 20, 20)

#    model = BiFusion(256, 256)
#    z = model(image1, image2)
#    print(z.size())

#**************************** DECODER *************************************#

#if __name__ == "__main__":
#    image = torch.rand(16, 1, 400, 400)
#    model = Decoder(0.5, 0.3, 0.2)
#    x = model(image)
#    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#    print(pytorch_total_params)
#    print(x.size())