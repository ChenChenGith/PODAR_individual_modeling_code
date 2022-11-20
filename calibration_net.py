import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt

class DataReader(Dataset):
    def __init__(self, subID: int) -> None:
        super().__init__()
        with open(r'data\dataset.pkl', 'rb') as f:
            self.data = pickle.load(f)[subID]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

def co_fn(data_dict, type='obj'):
    delta_v = [data['delta_v'] for data in data_dict]
    abs_v = [data['abs_v'] for data in data_dict]
    t_cur = [data['i'] for data in data_dict]
    d_cur = [data['d'] for data in data_dict]
    goal = 'st_angle' if type == 'obj' else 'response' # 'response', 'st_angle'
    label = [data[goal] for data in data_dict]
    # label = [data['response'] for data in data_dict]
    rows, cols = len(delta_v), delta_v[0].shape[0]

    return (torch.cat(delta_v).reshape(rows, cols).float(),
            torch.cat(abs_v).reshape(rows, cols).float(),
            torch.cat(t_cur).reshape(rows, cols).float(),
            torch.cat(d_cur).reshape(rows, cols).float(),
            torch.cat(label).reshape(rows).float())

class PODAR(nn.Module):
    def __init__(self, horizon=None) -> None:
        super(PODAR, self).__init__()
        self.m_ego, self.m_obj = torch.tensor(1.8), torch.tensor(1.8)
        self.alpha = nn.Parameter(torch.FloatTensor([0.7]))
        self.A = nn.Parameter(torch.FloatTensor([0.3]))
        self.B = nn.Parameter(torch.FloatTensor([0.5]))
        self.horizon = int(horizon * 10)
        self.scale = nn.Parameter(torch.FloatTensor([0.5 / 11.25]))

    def forward(self, delta_v, abs_v, i, d):
        """
        delta_v: [delta_v] * 31
        abs_v: [abs_v] * 31
        i: [i] * 31
        d: [d] * 31
        """
        m = 0.5 * (self.m_ego + self.m_obj)

        self.A_ = torch.clamp(self.A, 0.17, 50)
        self.B_ = torch.clamp(self.B, 0., 50)
        self.Alpha_ = torch.tensor([0.7])
        v = self.Alpha_ * delta_v + (1-self.Alpha_) * abs_v

        w_i = torch.exp(-1 * self.A_ * i)
        w_d = torch.exp(-1 * self.B_ * d)

        damage = torch.mul(v, torch.abs(v)) * m * 0.001 * self.scale
        attenu = torch.mul(w_i, w_d)

        podar_t = torch.mul(damage, attenu)

        podar = torch.max(podar_t[:, :self.horizon], dim=1)[0]

        self.p_max = podar.max()

        return podar

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)
    

def train(subID, type='obj'):
    peak_num_angle_25 = [3, 7, 5, 4, 3, 3, 4, 5]
    peak_num_respons = [4, 7, 7, 6, 5, 4, 7, 7]
    print('=====subject ID = {}, type: {}'.format(subID, type))
    batch_size = 77
    learning_rate = 0.01

    hor = peak_num_angle_25[subID] if type=='obj' else peak_num_respons[subID]
    net = PODAR(horizon=hor)
    net.apply(init_weights)
    # net = torch.load(r'podar_calibration\net.pkl')
    data_set = DataReader(subID)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: co_fn(x, type))

    loss_all = 0
    loss_rec = []
    ite_num = 1

    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000, )
    loss = torch.tensor([1])
    
    while ite_num < 50000:
        for batch_id, (delta_v, abs_v, t_cur, d_cur, label) in enumerate(data_loader):
            y = net(delta_v, abs_v, t_cur, d_cur)
            loss = criteria(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            loss_all += float(loss)
            loss_rec.append(loss.item())
            ave_loss = loss_all / (ite_num + 1)

            ite_num += 1

            # if ite_num % 10000 == 0:
            #     print('ite_num: {}, loss: {}, parameters A: {}, B: {}, alpha: {}, scale: {}, p_max: {}'.format(ite_num, loss.item(), net.A_.item(), net.B_.item(), net.Alpha_.item(), net.scale.item(), net.p_max.item()))
            #     # print('grad: A: {}, B: {}, Alpha: {}, horizon: {}'.format(net.A.grad, net.B.grad, net.alpha.grad, net.horizon.grad))

    print('---final parameters A: {}, B: {}, alpha: {}, scale: {}, p_max: {}'.format(net.A_.item(), net.B_.item(), net.Alpha_.item(), net.scale.item(), net.p_max.item()))
    # print('---final loss: {}'.format(loss))

    with open(r'data\dataset.pkl', 'rb') as f:
        data = pickle.load(f)[subID]
    
    delta_v, abs_v, t_cur, d_cur, label = co_fn(data, type=type)
    net_risk = net(delta_v, abs_v, t_cur, d_cur)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title('subject ID: {} - Training loss'.format(subID))
    plt.plot(loss_rec)
    plt.text(0.25, 0.5, 'A: {:.5f} \nB: {:.5f} \nAlpha: {:.5f} \nscale: {:.5f} \np_max: {:.3f} \nLoss: {:.5f}'.format(net.A.item(), net.B.item(), net.alpha.item(), net.scale.item(), net.p_max.item(), loss.item()), transform=plt.gca().transAxes)
    plt.subplot(122)
    plt.title('subject ID: {} - Verification'.format(subID))
    ln1, = plt.plot(net_risk.detach().numpy(), c='blue')
    plt.twinx()
    ln2, = plt.plot(label.detach().numpy(), c='r')

    if type == 'obj':
        plt.legend([ln1, ln2], ['PODAR estimated risk', 'Objective signal (steering angle)'])
        plt.savefig(r'trainning results\angle_{}.png'.format(subID))
        m = torch.jit.script(net)
        torch.jit.save(m, r'trainning results\angle_{}.pt'.format(subID))
    else:
        plt.legend([ln1, ln2], ['PODAR estimated risk', 'Subjective signal (oral response)'])
        plt.savefig(r'trainning results\response_{}.png'.format(subID))
        m = torch.jit.script(net)
        torch.jit.save(m, r'trainning results\response_{}.pt'.format(subID))

    # plt.show()

    # torch.onnx.export(net, (delta_v, abs_v, t_cur, d_cur), 'viz.pt', opset_version=11)
    # netron.start('viz.pt')
    return net.A_.item(), net.B_.item(), net.p_max.item(), net.scale.item()

def verify(subID):
    net = torch.load(r'trainning results\response_{}.pt'.format(subID))
    print(net.A, net.B, net.scale)
    with open(r'data\dataset.pkl', 'rb') as f:
        data = pickle.load(f)[subID]
    delta_v, abs_v, t_cur, d_cur, label, risk = co_fn(data, type='sub')
    net_risk = net(delta_v, abs_v, t_cur, d_cur)
    plt.figure(figsize=(7,7))
    plt.title('subject ID: {} - Verification'.format(subID))
    ln1, = plt.plot(net_risk.detach().numpy(), c='blue')
    plt.twinx()
    ln2, = plt.plot(label.detach().numpy(), c='r')
    plt.legend([ln1, ln2], ['PODAR estimated risk', 'Objective signal (steering angle)'])
    plt.show()
    ...

def calculate_R2(subID):

    folder_n = 'trainning results'
    net = torch.load(r'{}\angle_{}.pt'.format(folder_n, subID))
    with open(r'data\dataset.pkl', 'rb') as f:
        data = pickle.load(f)[subID]
    delta_v, abs_v, t_cur, d_cur, label = co_fn(data, type='obj')
    net_risk = net(delta_v, abs_v, t_cur, d_cur)
    
    label, net_risk = label.detach().numpy(), net_risk.detach().numpy()
    label_mean = np.mean(label)
    sst = np.power((label - label_mean), 2).sum()
    ssr = np.power((net_risk - label), 2).sum()
    r2 = 1 - ssr / sst

    net = torch.load(r'{}\response_{}.pt'.format(folder_n, subID))
    with open(r'data\dataset.pkl', 'rb') as f:
        data = pickle.load(f)[subID]
    delta_v, abs_v, t_cur, d_cur, label = co_fn(data, type='sub')
    net_risk = net(delta_v, abs_v, t_cur, d_cur)

    label, net_risk = label.detach().numpy(), net_risk.detach().numpy()
    label_mean = np.mean(label)
    sst = np.power((label - label_mean), 2).sum()
    ssr = np.power((net_risk - label), 2).sum()
    r2_ = 1 - ssr / sst

    return r2, r2_
    