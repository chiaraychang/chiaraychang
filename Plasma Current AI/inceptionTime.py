import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionTime(nn.Module):
    def __init__(self, time_steps, nb_classes, nb_filters=32, use_residual=True, 
                 depth=6, kernel_size=41):
        super(InceptionTime,self).__init__()
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.nb_classes = nb_classes
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32
        self.kernel_size_s = [self.kernel_size // (2 ** i) - 1 for i in range(3)]
        
        self.blocks = nn.ModuleList()
        for b in range(self.depth):
            in_maps = 1 if b==0 else self.nb_filters * 4

            self.blocks.append(inception_module(self.bottleneck_size, self.nb_filters,
                                                self.kernel_size_s, in_maps, time_steps))
        
        self.shortcuts = nn.ModuleList()
        for sc in range(self.depth//3):
            in_maps = 1 if sc==0 else self.nb_filters * 4

            self.shortcuts.append(shortcut_layer(in_maps, self.nb_filters * 4, time_steps))
        '''
        self.time_steps = time_steps
        self.conv_sc_1 = nn.Conv2d(1, self.nb_filters * 4, (1, 1), bias=False)
        self.bn_sc_1 = nn.BatchNorm2d(self.nb_filters * 4)
        self.conv_sc_2 = nn.Conv2d(self.nb_filters * 4, self.nb_filters * 4, (1, 1), bias=False)
        self.bn_sc_2 = nn.BatchNorm2d(self.nb_filters * 4)
        '''
        self.gap_layer = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(self.nb_filters * 4, nb_classes)
    
    def forward(self, x: torch.Tensor):

        input_res = x

        for d, block in enumerate(self.blocks):

            x = block(x)

            if self.use_residual and d % 3 == 2:
                '''
                if d // 3 == 0:
                    input_res = input_res.view(-1, 1, self.time_steps, 1)
                    shortcut_y = self.bn_sc_1(self.conv_sc_1(input_res))
                    x = shortcut_y + x
                    x = F.relu(x)
                else:
                    input_res = input_res.view(-1, self.nb_filters * 4, self.time_steps, 1)
                    shortcut_y = self.bn_sc_2(self.conv_sc_2(input_res))
                    x = shortcut_y + x
                    x = F.relu(x)
                '''
                x = self.shortcuts[d//3](input_res, x)
                
                input_res = x
                
        x = x.squeeze(3)
        x = self.gap_layer(x)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        x = F.log_softmax(x,1)

        return x.view(-1,self.nb_classes)


class shortcut_layer(nn.Module):
    def __init__(self, in_maps, out_maps, time_steps):
        super(shortcut_layer,self).__init__()
        self.in_maps = in_maps
        self.time_steps = time_steps
        self.conv = nn.Conv2d(in_maps, out_maps, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_maps)

    def forward(self, input_tensor, output_tensor):
        input_tensor = input_tensor.view(-1, self.in_maps, self.time_steps, 1)
        shortcut_y = self.bn(self.conv(input_tensor))
        x = shortcut_y + output_tensor
        x = F.relu(x)
        return x


class inception_module(nn.Module):
    def __init__(self, bottleneck_size, nb_filters, kernel_size_s, in_maps, time_steps):
        super(inception_module,self).__init__()
        self.in_maps  = in_maps
        self.time_steps = time_steps
        
        self.conv1 = nn.Conv2d(self.in_maps, bottleneck_size, (1, 1), bias=False)

        in_maps = bottleneck_size if in_maps>1 else in_maps
        self.conv2 = nn.Conv2d(in_maps, nb_filters, (kernel_size_s[0], 1), 1, ((kernel_size_s[0]+1)//2-1,0), bias=False)
        self.conv3 = nn.Conv2d(in_maps, nb_filters, (kernel_size_s[1], 1), 1, ((kernel_size_s[1]+1)//2-1,0), bias=False)
        self.conv4 = nn.Conv2d(in_maps, nb_filters, (kernel_size_s[2], 1), 1, ((kernel_size_s[2]+1)//2-1,0), bias=False)

        self.max_pool = nn.MaxPool2d((3, 1), stride=1, padding=(1, 0))
        self.conv5 = nn.Conv2d(in_maps, nb_filters, (1, 1), bias=False)

        self.bn = nn.BatchNorm2d(nb_filters*4)

        
    def forward(self, x):
        x = x.view(-1, self.in_maps, self.time_steps, 1)

        if self.in_maps > 1:
            x1 = self.conv1(x)
        else:
            x1 = x

        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        x4 = self.conv4(x1)
        x5 = self.conv5(self.max_pool(x1))

        x = torch.cat([x2, x3, x4, x5], axis=1)
        x = F.relu(self.bn(x))

        return x