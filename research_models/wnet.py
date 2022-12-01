import sys
sys.path.append('../')
from pycore.tikzeng import *
import copy


def _unet(ini_pos,in_ch, out_ch):
    unet = [
        to_Conv("conv1", 224, 64, offset=f"({ini_pos},0,0)", height=64, depth=64, width=4),
        to_Conv("conv2", 224, 64, offset=f"({ini_pos + 1},0,0)", height=64, depth=64, width=4),
        to_Pool("pool1", offset=f"({ini_pos + 2},0,0)", height=32, depth=32, width=1),

        to_Conv("conv3", 112, 128, offset=f"({ini_pos + 2.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv4", 112, 128, offset=f"({ini_pos + 3.5},0,0)", height=32, depth=32, width=4),
        to_Pool("pool2", offset=f"({ini_pos + 4.5},0,0)", height=16, depth=16, width=1),

        to_Conv("conv5", 64, 256, offset=f"({ini_pos + 5},0,0)", height=16, depth=16, width=4),
        to_Conv("conv6", 64, 256, offset=f"({ini_pos + 6},0,0)", height=16, depth=16, width=4),
        to_Pool("pool3", offset=f"({ini_pos + 7},0,0)", height=8, depth=8, width=1),

        to_Conv("conv7", 28, 512, offset=f"({ini_pos + 7.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv8", 28, 512, offset=f"({ini_pos + 8.5},0,0)", height=8, depth=8, width=4),
        to_Pool("pool4", offset=f"({ini_pos + 9.5},0,0)", height=4, depth=4, width=1),

        to_Conv("conv9", 14, 1024, offset=f"({ini_pos + 10},0,0)", height=4, depth=4, width=4),
        to_Conv("conv10", 14, 1024, offset=f"({ini_pos + 11},0,0)", height=4, depth=4, width=4),
        to_UnPool("unpool1", offset=f"({ini_pos + 12},0,0)", height=8, depth=8, width=1),

        to_ConvRes("con_res_1", 28, 512, offset=f"({ini_pos + 12.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv11", 28, 512, offset=f"({ini_pos + 13.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv12", 28, 512, offset=f"({ini_pos + 14.5},0,0)", height=8, depth=8, width=4),
        to_UnPool("unpool2", offset=f"({ini_pos + 15.5},0,0)", height=16, depth=16, width=1),
        to_skip(of='conv8', to='con_res_1'),

        to_ConvRes("con_res_2", 64, 256, offset=f"({ini_pos + 16},0,0)", height=16, depth=16, width=4),
        to_Conv("conv13", 64, 256, offset=f"({ini_pos + 17},0,0)", height=16, depth=16, width=4),
        to_Conv("conv14", 64, 256, offset=f"({ini_pos + 18},0,0)", height=16, depth=16, width=4),
        to_UnPool("unpool3", offset=f"({ini_pos + 19},0,0)", height=32, depth=32, width=1),
        to_skip(of='conv6', to='con_res_2'),

        to_ConvRes("con_res_3", 112, 128, offset=f"({ini_pos + 19.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv15", 112, 128, offset=f"({ini_pos + 20.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv16", 112, 128, offset=f"({ini_pos + 21.5},0,0)", height=32, depth=32, width=4),
        to_UnPool("unpool4", offset=f"({ini_pos + 22.5},0,0)", height=64, depth=64, width=1),
        to_skip(of='conv4', to='con_res_3'),

        to_ConvRes("con_res_4", 224, 64, offset=f"({ini_pos + 23},0,0)", height=64, depth=64, width=4),
        to_Conv("conv17", 224, 64, offset=f"({ini_pos+ 24},0,0)", height=64, depth=64, width=4),
        to_Conv("conv18", 224, 64, offset=f"({ini_pos + 25},0,0)", height=64, depth=64, width=4),
        to_skip(of='conv2', to='con_res_4'),


        to_Conv("conv19", 224, out_ch, offset=f"({ini_pos + 26},0,0)", height=64, depth=64, width=2),
    ]
    last_pos = ini_pos + 26
    return unet, last_pos

def main():
    begin = [to_head( '..' ),
             to_cor(),
             to_begin()]
    end = [to_end()]

    encoder, last_enc_pos = _unet(0, 3, 2)
    softmax = to_SoftMax("softmax1",s_filer=2, offset=f"({last_enc_pos+1}, 0, 0)", height=64, depth=64, width = 2)
    decoder, last_dec_pos = _unet(last_enc_pos + 1.5, 1, 3)
    wnet = begin + encoder + [softmax] + decoder + end

    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(wnet, namefile + '.tex' )

if __name__ == '__main__':
    main()
