from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""
加上了这四行之后，即使用python2，也可以正常的跑这段在python3之下的程序。
absolute_import引入了python3里才有的import行为，即import一个模块会去引用标准库，而不是当前目录下的同名文件。
division引入了python3的除法，即/代表精准除，//代表了floor除法。
print_function就是python3的print是必须当做函数调用的
为了适应Python 3.x的新的字符串的表示方法，在2.7版本的代码中，可以通过unicode_literals来使用Python 3.x的新的语法
"""
import argparse
import numpy as np
import os
import pandas as pd
from util import StandardScaler
import torch
import gc
from util import normalize_features


# data_dir = 'data\\original_data'
# data_fp = os.path.join(data_dir, 'metr-la.h5') #加载csv文件  parking_2month.csv
#
# data = pd.read_csv(data_fp, header=None).values
cvae_model = torch.load("./cvae_models/3convtest_wo_grus.pkl",map_location=torch.device('cpu'))
def get_augmented_features(hist_data):
    # num_samples = hist_data.shape[0]
    num_nodes = hist_data.shape[0]
    z = torch.randn([num_nodes, cvae_model.latent_size])
    augmented_features = cvae_model.inference(z,torch.tensor(data=hist_data, dtype=torch.float32)).detach()
    augmented_features = np.expand_dims(augmented_features.T.numpy(),axis=-1)
    return augmented_features

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, x_hist_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:输入数据的索引偏置
    :param y_offsets:输出数据的索引偏置
    :param add_time_in_day:将时间转为以天为单位
    :param add_day_in_week:
    :param scaler:
    :return:输入数据和输出数据9o
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    hist_len=len(x_hist_offsets)
    last_train_index=int((num_samples-hist_len+1)*0.7+(hist_len-1))
    scaler=StandardScaler(mean=df[:last_train_index,:].mean(),
                                     std=df[:last_train_index,:].std())
    data=scaler.transform(df)
    # print(f"num_sample:{num_samples} num_nodes:{num_nodes}")
    # 将二维数组[M,92]扩张成三维[M,92,1]
    # data = np.expand_dims(df.values, axis=-1)
    data = np.expand_dims(data, axis=-1).astype(np.float32)  # [34272,207]->[34272,207,1]
    df = np.expand_dims(df, axis=-1).astype(np.float32)  # [34272,207]->[34272,207,1]
    # feature_list = [data]
    # if add_time_in_day:
    #     time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1,
    #                                                                                             "D")  # .astype()转换时间维度，df.col / np.timedelta64(1,'D')去除单位
    #     time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))  # numpy.tile()重复数组，transpose就是对多维数组进行转置
    #     feature_list.append(time_in_day)
    # if add_day_in_week:
    #     dow = df.index.dayofweek
    #     dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
    #     feature_list.append(dow_tiled)
    #
    # data = np.concatenate(feature_list,
    #                       axis=-1)  # np.concatenate([a,b],axis=-1)表示在第三个中括号([[[......]]]从外到内，一次为第一个中括号、第二个、第三个......)上添加元素。
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x_aug, x, y = [], [], []
    # x_aug,x,y=np.array([]),np.array([]),np.array([])
    min_t = abs(min(x_hist_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        print("t:{:d}".format(t))
        aug_f=get_augmented_features(np.squeeze(data[t + x_hist_offsets, ...], axis=-1).T)
        x_aug.append(aug_f[-12:,:,:])
        x.append(df[t + x_offsets, ...])
        y.append(df[t + y_offsets, ...])
        # x_aug=np.append(x_aug,aug_f)
        # x=np.append(x,data[t + x_offsets, ...])
        # y=np.append(y,data[t + y_offsets, ...])
    x_aug = np.stack(x_aug, axis=0)  # 堆叠的多个数组
    x = np.stack(x, axis=0)  # 堆叠的多个数组
    y = np.stack(y, axis=0)
    # x_aug1, x1, y1 = [], [], []
    # for t in range(17130, max_t):  # t is the index of the last observation.
    #     print("t:{:d}".format(t))
    #     aug_f=get_augmented_features(np.squeeze(data[t + x_hist_offsets, ...], axis=-1).T)
    #     x_aug1.append(aug_f)
    #     x1.append(data[t + x_offsets, ...])
    #     y1.append(data[t + y_offsets, ...])
    #     # x_aug=np.append(x_aug,aug_f)
    #     # x=np.append(x,data[t + x_offsets, ...])
    #     # y=np.append(y,data[t + y_offsets, ...])
    # x_aug1 = np.stack(x_aug1, axis=0)  # 堆叠的多个数组
    # x1 = np.stack(x1, axis=0)  # 堆叠的多个数组
    # y1 = np.stack(y1, axis=0)
    # x_aug=np.concatenate((x_aug,x_aug1),axis=0)
    # x=np.concatenate((x,x1),axis=0)
    # y=np.concatenate((y,y1),axis=0)
    # del x_aug1,x1,y1
    # gc.collect()
    print(f"x_aug:{x_aug.shape}, x:{x.shape}, y:{y.shape}")
    return x_aug, x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y, seq_length_x_hist = args.seq_length_x, args.seq_length_y, args.seq_length_x_hist
    # df = pd.read_hdf(args.traffic_df_filename)

    df = pd.read_hdf(args.traffic_df_filename).to_numpy()  # data
    # df=np.load(args.traffic_df_filename)
    # print(df.shape)
    # df=df.T

    # df=features
    # df_aug=augmented_features

    # 0 is the latest observed sample.
    x_hist_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x_hist - 1), 1, 1),)))
    print(x_hist_offsets)
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    print(x_offsets)
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x_aug, x, y = generate_graph_seq2seq_io_data(
        df[:17186,:],
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        x_hist_offsets=x_hist_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )
    x_aug1, x1, y1 = generate_graph_seq2seq_io_data(
        df[15171:, :],
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        x_hist_offsets=x_hist_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )
    # x_aug, y_aug = generate_graph_seq2seq_io_data(
    #     df_aug,
    #     x_offsets=x_offsets,
    #     y_offsets=y_offsets,
    #     add_time_in_day=None,
    #     add_day_in_week=args.dow,
    # )
    x=np.concatenate((x, x1), axis=0)
    x_aug=np.concatenate((x_aug, x_aug1), axis=0)
    y=np.concatenate((y, y1), axis=0)
    del x1,x_aug1,y1
    gc.collect()

    print("x_hist shape: ", x_aug.shape, "x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    # 利用numpy保存数据进.npz文件中，每个文件中都包含输入数据、标签数据、输入数据索引偏置、标签数据索引偏置，
    # 和上面说的差不多，只是形状变成了[12,1]的二维数组
    num_samples = x_aug.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_aug_train, x_train, y_train = x_aug[:num_train], x[:num_train], y[:num_train]
    print(f"x_train shape:{x_train.shape}")
    # val
    x_aug_val, x_val, y_val = (
        x_aug[num_train: num_train + num_val],
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_aug_test, x_test, y_test = x_aug[-num_test:], x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x_aug, _x, _y = locals()["x_aug_" + cat], locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x_hist: ", _x_aug.shape, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x_aug=_x_aug,
            x=_x,
            y=_y,
            x_hist_offsets=x_hist_offsets.reshape(list(x_hist_offsets.shape) + [1]),
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/temporal_data/metr-la-12-with_hist_wo_gru",
                        help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/original_data/metr-la.h5",
                        help="Raw traffic readings.", )
    # parser.add_argument("--traffic_df_filename", type=str, default="data/cvae_features/original-gaussiandiffusion-vae.npy",
    #                     help="Raw traffic readings.", )
    # parser.add_argument("--augmented_features", type=str,
    #                     default="data/cvae_features/original-gaussiandiffusion-edropout-econv.npy",
    #                     help="Augmented features", )
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--seq_length_x_hist", type=int, default=2016, help="Sequence Length.", )
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true', )

    args = parser.parse_args()
    # features=pd.read_hdf(args.traffic_df_filename)
    # features=np.load(args.traffic_df_filename)
    # features=features.T
    # print(features.shape)
    # features=features.columns[0:]
    # for col in features.columns[0:]:  # 忽略第一列时间戳
    #     max_val = features[col].max()
    #     min_val = features[col].min()
    #     range_val = max_val - min_val
    #     norm_col = (features[col] - min_val) / range_val
    #     features[col] = norm_col
    # augmented_features=np.load(args.augmented_features)
    # augmented_features=augmented_features.T
    generate_train_val_test(args)

"""
argparse是一个Python模块：命令行选项、参数和子命令解析器。
主要有三个步骤：
创建 ArgumentParser() 对象
调用 add_argument() 方法添加参数
使用 parse_args() 解析添加的参数
"""
