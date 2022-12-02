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

        to_Conv("conv5", 56, 256, offset=f"({ini_pos + 5},0,0)", height=16, depth=16, width=4),
        to_Conv("conv6", 56, 256, offset=f"({ini_pos + 6},0,0)", height=16, depth=16, width=4),
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

        to_ConvRes("con_res_2", 56, 256, offset=f"({ini_pos + 16},0,0)", height=16, depth=16, width=4),
        to_Conv("conv13", 56, 256, offset=f"({ini_pos + 17},0,0)", height=16, depth=16, width=4),
        to_Conv("conv14", 56, 256, offset=f"({ini_pos + 18},0,0)", height=16, depth=16, width=4),
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

        to_Conv("conv19", 224, out_ch, offset=f"({ini_pos + 26},0,0)", height=64, depth=64, width=2, color="\OneConvColor", caption=f"{out_ch}D")
    ]
    last_pos = ini_pos + 26
    return unet, last_pos

def _unet2(ini_pos,in_ch, out_ch):
    unet = [
        to_Conv("conv1", 224, 64, offset=f"({ini_pos},0,0)", height=64, depth=64, width=4),
        to_Conv("conv2", 224, 64, offset=f"({ini_pos + 1},0,0)", height=64, depth=64, width=4),
        to_Pool("pool1", offset=f"({ini_pos + 2},0,0)", height=32, depth=32, width=1),

        to_Conv("conv3", 112, 128, offset=f"({ini_pos + 2.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv4", 112, 128, offset=f"({ini_pos + 3.5},0,0)", height=32, depth=32, width=4),
        to_Pool("pool2", offset=f"({ini_pos + 4.5},0,0)", height=16, depth=16, width=1),

        to_Conv("conv5", 56, 256, offset=f"({ini_pos + 5},0,0)", height=16, depth=16, width=4),
        to_Conv("conv6", 56, 256, offset=f"({ini_pos + 6},0,0)", height=16, depth=16, width=4),
        to_Conv("conv7", 56, 256, offset=f"({ini_pos + 7},0,0)", height=16, depth=16, width=4),
        to_Pool("pool3", offset=f"({ini_pos + 8},0,0)", height=8, depth=8, width=1),

        to_Conv("conv8", 28, 512, offset=f"({ini_pos + 8.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv9", 28, 512, offset=f"({ini_pos + 9.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv10", 28, 512, offset=f"({ini_pos + 10.5},0,0)", height=8, depth=8, width=4),
        to_Pool("pool4", offset=f"({ini_pos + 11.5},0,0)", height=4, depth=4, width=1),

        # new one
        to_Conv("conv11", 14, 1024, offset=f"({ini_pos + 12},0,0)", height=4, depth=4, width=4),
        to_Conv("conv12", 14, 1024, offset=f"({ini_pos + 13},0,0)", height=4, depth=4, width=4),
        to_Conv("conv13", 14, 1024, offset=f"({ini_pos + 14},0,0)", height=4, depth=4, width=4),
        to_Pool("pool5", offset=f"({ini_pos + 15},0,0)", height=3, depth=3, width=1),

        to_Conv("conv14", 7, 2048, offset=f"({ini_pos + 15.5},0,0)", height=3, depth=3, width=4),
        to_Conv("conv15", 7, 2048, offset=f"({ini_pos + 16.5},0,0)", height=3, depth=3, width=4),
        to_UnPool("unpool1", offset=f"({ini_pos + 17.5},0,0)", height=4, depth=4, width=1),

        # new one
        to_ConvRes("con_res_1", 28, 1024, offset=f"({ini_pos + 18},0,0)", height=4, depth=4, width=4),
        to_Conv("conv16", 14, 1024, offset=f"({ini_pos + 19},0,0)", height=4, depth=4, width=4),
        to_Conv("conv17", 14, 1024, offset=f"({ini_pos + 20},0,0)", height=4, depth=4, width=4),
        to_Conv("conv18", 14, 1024, offset=f"({ini_pos + 21},0,0)", height=4, depth=4, width=4),
        to_UnPool("unpool2", offset=f"({ini_pos + 22},0,0)", height=8, depth=8, width=1),
        to_skip(of='conv13', to='con_res_1'),

        to_ConvRes("con_res_2", 28, 512, offset=f"({ini_pos + 22.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv19", 28, 512, offset=f"({ini_pos + 23.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv20", 28, 512, offset=f"({ini_pos + 24.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv21", 28, 512, offset=f"({ini_pos + 25.5},0,0)", height=8, depth=8, width=4),
        to_UnPool("unpool3", offset=f"({ini_pos + 26.5},0,0)", height=16, depth=16, width=1),
        to_skip(of='conv10', to='con_res_2'),

        to_ConvRes("con_res_3", 56, 256, offset=f"({ini_pos + 27},0,0)", height=16, depth=16, width=4),
        to_Conv("conv22", 56, 256, offset=f"({ini_pos + 28},0,0)", height=16, depth=16, width=4),
        to_Conv("conv23", 56, 256, offset=f"({ini_pos + 29},0,0)", height=16, depth=16, width=4),
        to_Conv("conv24", 56, 256, offset=f"({ini_pos + 30},0,0)", height=16, depth=16, width=4),
        to_UnPool("unpool4", offset=f"({ini_pos + 31},0,0)", height=32, depth=32, width=1),
        to_skip(of='conv7', to='con_res_3'),

        to_ConvRes("con_res_4", 112, 128, offset=f"({ini_pos + 31.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv25", 112, 128, offset=f"({ini_pos + 32.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv26", 112, 128, offset=f"({ini_pos + 33.5},0,0)", height=32, depth=32, width=4),
        to_UnPool("unpool5", offset=f"({ini_pos + 34.5},0,0)", height=64, depth=64, width=1),
        to_skip(of='conv4', to='con_res_4'),

        to_ConvRes("con_res_5", 224, 64, offset=f"({ini_pos + 35},0,0)", height=64, depth=64, width=4),
        to_Conv("conv27", 224, 64, offset=f"({ini_pos+ 36},0,0)", height=64, depth=64, width=4),
        to_Conv("conv28", 224, 64, offset=f"({ini_pos + 37},0,0)", height=64, depth=64, width=4),
        to_skip(of='conv2', to='con_res_5'),

        to_Conv("conv29", 224, out_ch, offset=f"({ini_pos + 38},0,0)", height=64, depth=64, width=2, color="\OneConvColor", caption=f"{out_ch}D")
    ]
    last_pos = ini_pos + 38
    return unet, last_pos

def _unet3(ini_pos,in_ch, out_ch):
    unet = [
        to_Conv("conv1", 224, 64, offset=f"({ini_pos},0,0)", height=64, depth=64, width=4),
        to_Conv("conv2", 224, 64, offset=f"({ini_pos + 1},0,0)", height=64, depth=64, width=4),
        to_Pool("pool1", offset=f"({ini_pos + 2},0,0)", height=32, depth=32, width=1),

        to_Conv("conv3", 112, 128, offset=f"({ini_pos + 2.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv4", 112, 128, offset=f"({ini_pos + 3.5},0,0)", height=32, depth=32, width=4),
        to_Pool("pool2", offset=f"({ini_pos + 4.5},0,0)", height=16, depth=16, width=1),

        to_Conv("conv5", 56, 256, offset=f"({ini_pos + 5},0,0)", height=16, depth=16, width=4),
        to_Conv("conv6", 56, 256, offset=f"({ini_pos + 6},0,0)", height=16, depth=16, width=4),
        to_Conv("conv7", 56, 256, offset=f"({ini_pos + 7},0,0)", height=16, depth=16, width=4),
        to_Pool("pool3", offset=f"({ini_pos + 8},0,0)", height=8, depth=8, width=1),

        to_Conv("conv8", 28, 512, offset=f"({ini_pos + 8.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv9", 28, 512, offset=f"({ini_pos + 9.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv10", 28, 512, offset=f"({ini_pos + 10.5},0,0)", height=8, depth=8, width=4),
        to_Pool("pool4", offset=f"({ini_pos + 11.5},0,0)", height=4, depth=4, width=1),

        to_Conv("conv11", 14, 1024, offset=f"({ini_pos + 12},0,0)", height=4, depth=4, width=4),
        to_Conv("conv12", 14, 1024, offset=f"({ini_pos + 13},0,0)", height=4, depth=4, width=4),
        to_UnPool("unpool1", offset=f"({ini_pos + 14},0,0)", height=8, depth=8, width=1),

        to_ConvRes("con_res_1", 28, 512, offset=f"({ini_pos + 14.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv13", 28, 512, offset=f"({ini_pos + 15.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv14", 28, 512, offset=f"({ini_pos + 16.5},0,0)", height=8, depth=8, width=4),
        to_Conv("conv15", 28, 512, offset=f"({ini_pos + 17.5},0,0)", height=8, depth=8, width=4),
        to_UnPool("unpool2", offset=f"({ini_pos + 18.5},0,0)", height=16, depth=16, width=1),
        to_skip(of='conv10', to='con_res_1'),

        to_ConvRes("con_res_2", 56, 256, offset=f"({ini_pos + 19},0,0)", height=16, depth=16, width=4),
        to_Conv("conv16", 56, 256, offset=f"({ini_pos + 20},0,0)", height=16, depth=16, width=4),
        to_Conv("conv17", 56, 256, offset=f"({ini_pos + 21},0,0)", height=16, depth=16, width=4),
        to_Conv("conv18", 56, 256, offset=f"({ini_pos + 22},0,0)", height=16, depth=16, width=4),
        to_UnPool("unpool3", offset=f"({ini_pos + 23},0,0)", height=32, depth=32, width=1),
        to_skip(of='conv7', to='con_res_2'),

        to_ConvRes("con_res_3", 112, 128, offset=f"({ini_pos + 23.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv19", 112, 128, offset=f"({ini_pos + 24.5},0,0)", height=32, depth=32, width=4),
        to_Conv("conv20", 112, 128, offset=f"({ini_pos + 25.5},0,0)", height=32, depth=32, width=4),
        to_UnPool("unpool4", offset=f"({ini_pos + 26.5},0,0)", height=64, depth=64, width=1),
        to_skip(of='conv4', to='con_res_3'),

        to_ConvRes("con_res_4", 224, 64, offset=f"({ini_pos + 27},0,0)", height=64, depth=64, width=4),
        to_Conv("conv21", 224, 64, offset=f"({ini_pos+ 28},0,0)", height=64, depth=64, width=4),
        to_Conv("conv22", 224, 64, offset=f"({ini_pos + 29},0,0)", height=64, depth=64, width=4),
        to_skip(of='conv2', to='con_res_4'),

        to_Conv("conv19", 224, out_ch, offset=f"({ini_pos + 30},0,0)", height=64, depth=64, width=2, color="\OneConvColor", caption=f"{out_ch}D")
    ]
    last_pos = ini_pos + 30
    return unet, last_pos

def main():
    begin = [to_head( '..' ),
             to_cor(),
             to_begin()]
    end = [to_end()]

    # encoder, last_enc_pos = _unet(0, 3, 2)
    # softmax = to_SoftMax("softmax1",s_filer=2, offset=f"({last_enc_pos+1}, 0, 0)", height=64, depth=64, width = 2)
    # decoder, last_dec_pos = _unet(last_enc_pos + 1.5, 1, 3)
    # input = to_input( '../examples/satellite.png', height=12, width=12 )
    # wnet = begin + [input] + encoder + [softmax] + decoder + end
    #
    # namefile = str(sys.argv[0]).split('.')[0]
    # to_generate(wnet, namefile + '.tex' )

    #############################################################

    # encoder, last_enc_pos = _unet2(0, 3, 2)
    # softmax = to_SoftMax("softmax1", s_filer=2, offset=f"({last_enc_pos + 1}, 0, 0)", height=64, depth=64, width=2)
    # decoder, last_dec_pos = _unet2(last_enc_pos + 1.5, 1, 3)
    # input = to_input('../examples/satellite.png', height=12, width=12)
    # wnet = begin + [input] + encoder + [softmax] + decoder + end
    #
    # namefile = str(sys.argv[0]).split('.')[0]
    # to_generate(wnet, namefile + '.tex')

    #############################################################


    encoder, last_enc_pos = _unet3(0, 3, 2)
    softmax = to_SoftMax("softmax1", s_filer=2, offset=f"({last_enc_pos + 1}, 0, 0)", height=64, depth=64, width=2)
    decoder, last_dec_pos = _unet3(last_enc_pos + 1.5, 1, 3)
    input = to_input('../examples/satellite.png', height=12, width=12)
    wnet = begin + [input] + encoder + [softmax] + decoder + end

    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(wnet, namefile + '.tex')

if __name__ == '__main__':
    main()
