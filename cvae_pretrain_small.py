import sys
import gc
import random
import torch
import torch.optim as optim
from tqdm import tqdm, trange
from cvae_models import VAE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from util import *




# Training settings
exc_path = sys.path[0]


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

def slide_ts(input_tensor,window_size,stride):

    # 在第二个维度上滑动切分
    sliced_x = []
    for i in range(0, input_tensor.size(1) - window_size + 1, stride):
        sliced_x.append(input_tensor[:, i:i + window_size])

    # 将切分后的张量列表拼接成一个新的张量
    result_tensor = torch.cat(sliced_x, dim=0)
    return result_tensor



def generated_generator(args, device, adj_scipy,features):
    # scaler = StandardScaler(mean=features[...].mean(), std=features[...].std())
    # features_ld, pca_list, final_pca_list = [], [], []
    # for i in range(features.shape[0]):
    #     # 将数据按照窗口大小w=1000划分成子序列
    #     w = 24
    #     X_list = features[i].reshape(-1, 24)
    #     pca_list,final_pca_list=[],[]
    #     # 对每个子序列进行一维PCA降维
    #     pca = PCA(n_components=1)
    #     w_list = []
    #     for X in X_list:
    #         w_list.append(pca.fit_transform(X.reshape(-1, 1)).squeeze())
    #         pca_list.append(pca.components_)
    #
    #     # 将所有子序列的主成分向量沿着时间轴拼接起来构成矩阵
    #     W = np.vstack(w_list)
    #
    #     # 再次应用PCA进行降维
    #     final_pca = PCA(n_components=1, svd_solver='randomized')
    #     x_final = final_pca.fit_transform(W)
    #     features_ld.append(x_final)
    #     final_pca_list.append(final_pca.components_)
    # features=torch.tensor(np.array(features_ld).squeeze(-1)).to(torch.float32)
    x_list, c_list = [], []
    for i in trange(adj_scipy.shape[0]):
        neighbors_index = list(adj_scipy[i].nonzero()[1])  # 邻居节点的索引
        x = features[neighbors_index]  # 邻居节点的特征（num_neighb,num_feature)
        c = torch.tile(features[i], (x.shape[0], 1))  # torch.tile用于复制，复制中心节点特征，（num_neighb,num_feature)
        x_slided=slide_ts(x,2016,2016)
        c_slided=slide_ts(c,2016,2016)
        x_list.append(x_slided)  # x_list(num_nodes,num_neighb,num_feature)
        c_list.append(c_slided)  # c_list(num_nodes,num_neighb,num_feature)
    features_x = torch.vstack(x_list)  # 减少无用维度
    features_c = torch.vstack(c_list)
    del x_list
    del c_list
    gc.collect()  # 垃圾回收

    cvae_dataset = TensorDataset(features_x, features_c)
    # # 从数据集中随机选择200个样本
    # num_selected_samples = 50000
    # selected_samples_indices = random.sample(range(features_x.shape[0]), num_selected_samples)
    #
    # # 将选择出来的样本按照7:3的比例分成训练集和测试集
    # num_train_samples = int(0.7 * num_selected_samples)
    # train_indices = selected_samples_indices[:num_train_samples]
    # test_indices = selected_samples_indices[num_train_samples:]
    #
    # # 创建训练集和测试集的数据加载器
    # train_loader = torch.utils.data.DataLoader(cvae_dataset, batch_size=args.batch_size,
    #                                            sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    # test_loader = torch.utils.data.DataLoader(cvae_dataset, batch_size=args.batch_size,
    #                                           sampler=torch.utils.data.SubsetRandomSampler(test_indices))
    # 划分数据集为训练集和验证集（70%训练集，30%验证集）
    train_size = int(0.7 * len(cvae_dataset))
    val_size = len(cvae_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(cvae_dataset, [train_size, val_size])
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 训练集加载器
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)  # 验证集加载器
    # cvae_dataset_dataloader = DataLoader(cvae_dataset,  batch_size=args.batch_size)
    
    # Pretrain
    cvae = VAE(encoder_layer_sizes=[int(features_x.shape[1]), 256],
               latent_size=args.latent_size, 
               decoder_layer_sizes=[256,int(features_x.shape[1])],
               conditional=args.conditional, 
               conditional_size=int(features_x.shape[1]),

               ).to(device)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)
    # cvae_optimizer=tf.optimizers.Adagrad(learning_rate=args.pretrain_lr, initial_accumulator_value=0.1)

    # Pretrain
    for epoch in trange(100, desc='Run CVAE Train'):
        for batch, (x, c) in enumerate(tqdm(train_loader)):
            cvae.train() # 启动模型训练过程

            x, c = x.to(device), c.to(device)
            if args.conditional:
                recon_x, mean, log_var, _ = cvae(x, c)
            else:
                recon_x, mean, log_var, _ = cvae(x)
            cvae_loss = loss_fn(recon_x, x, mean, log_var)

            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()
            # 在验证集上评估模型
        cvae.eval()
        with torch.no_grad():
            cvae_loss = 0
            total_samples = 0
            for val_batch_idx, (x, c) in enumerate(val_loader):
                x, c = x.to(device), c.to(device)
                if args.conditional:
                    recon_x, mean, log_var, _ = cvae(x, c)
                else:
                    recon_x, mean, log_var, _ = cvae(x)
                cvae_loss += loss_fn(recon_x, x, mean, log_var)
                total_samples+=1
            average_loss=cvae_loss/total_samples
            print(f'Epoch {epoch + 1}, Validation Loss: {average_loss:.4f}')

        # 恢复模型为训练模式
        cvae.train()
    return cvae
