import tensorflow as tf

ngf = 32 # Number of filters in first layer of generator
ndf = 64 # Number of filters in first layer of discriminator
batch_size = 1 # batch_size
pool_size = 50 # pool_size
img_width = 256 # Imput image will of width 256
img_height = 256 # Input image will be of height 256
img_depth = 3 # RGB format


o_c1 = general_conv2d(input_gen,
       num_features=ngf,
       window_width=7,
       window_height=7,
       stride_width=1,
       stride_height=1)
def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(inputconv, num_features, [window_width, window_height], [stride_width, stride_height],
                                        padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))

def resnet_blocks(input_res, num_features):    #转换层

    out_res_1 = general_conv2d(input_res, num_features,
                               window_width=3,
                               window_heigth=3,
                               stride_width=1,
                               stride_heigth=1)
    out_res_2 = general_conv2d(out_res_1, num_features,
                               window_width=3,
                               window_heigth=3,
                               stride_width=1,
                               stride_heigth=1)
    return (out_res_2 + input_res)