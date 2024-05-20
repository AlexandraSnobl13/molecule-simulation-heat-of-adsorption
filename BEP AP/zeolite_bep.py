import os
import sys
sys.path.append(os.getcwd())
import argparse
import torch
import pickle
# from scipy.io import loadmat
import torch.nn as nn
import numpy as np
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import radius_graph
# import torch_sparse
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# import statistics
# from math import sqrt
from tqdm import tqdm
# from torch_scatter import scatter_add
from schnet import SchNet
from schnet_zeolite import SchNetZeo
from setmodel import SetModel
from descriptor import DescriptorModel

def get_tensor(file):
    return torch.tensor(np.load(file))

def get_atoms(file):

    with open(file) as f:
        lines = f.readlines()
    lines = [i.strip().split() for i in lines]
    lines = [i for i in lines if len(i)>1]


    at_pos = [i[1:5] for i in lines if i[1] in ['Si', 'Al']]
    atom = np.array([1 if i[0]=='Al' else 0 for i in at_pos])
    X = np.array([list(map(float, i[1:])) for i in at_pos])

    at_pos_O = [i[1:5] for i in lines if i[1] == 'O']
    X_o = np.array([list(map(float, i[1:])) for i in at_pos_O])
    return atom, X, X_o

def periodic_boundary(d):
    '''
    Applies periodic boundary conditions to the difference vector d (fractional coordinates)
    '''
    
    d = torch.where(d<-0.5, d+1, d)
    d = torch.where(d>0.5, d-1, d)
    
    return d

def get_transform_matrix(a, b, c, alpha, beta, gamma):
    """
    a, b, c: lattice vector lengths (angstroms)
    alpha, beta, gamma: lattice vector angles (degrees)

    Returns the transformation matrix from fractional to cartesian coordinates
    """
    # convert to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    zeta = (np.cos(alpha) - np.cos(gamma) * np.cos(beta))/np.sin(gamma)
    
    h = np.zeros((3,3))
    
    h[0,0] = a
    h[0,1] = b * np.cos(gamma)
    h[0,2] = c * np.cos(beta)

    h[1,1] = b * np.sin(gamma)
    h[1,2] = c * zeta

    h[2,2] = c * np.sqrt(1 - np.cos(beta)**2 - zeta**2)

    return h

def fractional_to_cartesian(X, h):
    '''
    X: (N, 3) tensor
    h: (3, 3) tensor
    '''
    return torch.matmul(X, h.T)

def cartesian_to_fractional(X, h):
    '''
    X: (N, 3) tensor
    h: (3, 3) tensor
    '''
    return torch.matmul(X, torch.inverse(h).T)


def get_distance(X1, X2, h):
    '''
    Calculates pairwise distance between X1 and X2
    X1, X2: (N, 3) tensor (fractional coordinates)
    h: (3, 3) tensor

    Returns a (N,) tensor
    '''

    d_ij = X1 - X2
    d_ij = periodic_boundary(d_ij)
    d_ij = fractional_to_cartesian(d_ij, h)
    d_ij = torch.norm(d_ij, dim=1)
    return d_ij


def get_distance_matrix(X1, X2, h):
    '''
    Calculates pairwise distance matrix between X1 and X2
    X1: (N, 3) tensor (fractional coordinates)
    X2: (M, 3) tensor (fractional coordinates)
    h: (3, 3) tensor

    Returns a (N, M) tensor
    '''

    d_ij = X1.unsqueeze(1) - X2
    d_ij = periodic_boundary(d_ij)
    d_ij = fractional_to_cartesian(d_ij, h)
    d_ij = torch.norm(d_ij, dim=2)
    return d_ij

def get_edge_index(X, X_o, h):
    '''
    Gets edge index for the atoms based on T-O-T bonds
    X: (N, 3) tensor
    X_o: (N*2, 3) tensor
    h: (3, 3) tensor

    Returns: (N, N) tensor
    '''

    # calculate distance between X and X_o
    d_t_o = get_distance_matrix(X, X_o, h)
    idx_i, idx_j = d_t_o.argsort(dim=0)[:2,]

    # create edge index
    idx_1 = torch.cat([idx_i, idx_j], dim=0)
    idx_2 = torch.cat([idx_j, idx_i], dim=0)
    edge_index = torch.stack([idx_1, idx_2], dim=0)
    
    return edge_index

def get_triplets(edge_index):
    """
    Calculates i,j,k triplets for T-O-T-O-T bonds

    edge_index: (2, M) tensor
    """
    n = edge_index.max().item() + 1

    ind_i, ind_j = edge_index

    value = torch.arange(ind_j.size(0), device=ind_j.device)
    adj_t = SparseTensor(row=ind_i, col=ind_j, value=value,
                            sparse_sizes=(n,n))
    adj_t_row = adj_t[ind_j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = ind_i.repeat_interleave(num_triplets)
    idx_j = ind_j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return idx_i, idx_j, idx_k, idx_kj, idx_ji

def get_angles(X, h, idx_i_1, idx_j_1, idx_k_1):
    """
    Calculates the angle between T-O-T-O-T bonds

    X: (N, 3) tensor
    h: (3, 3) tensor
    idx_i_1: (M,) tensor
    idx_j_1: (M,) tensor
    idx_k_1: (M,) tensor

    Returns: (M,) tensor
    """

    d_ji_1 = X[idx_j_1] - X[idx_i_1]
    d_ji_1 = periodic_boundary(d_ji_1)

    d_kj_1 = X[idx_k_1] - X[idx_j_1]
    d_kj_1 = periodic_boundary(d_kj_1)

    d_kj_1 = fractional_to_cartesian(d_kj_1, h)
    d_ji_1 = fractional_to_cartesian(d_ji_1, h)

    a = (d_ji_1*d_kj_1).sum(dim=1)
    b = torch.cross(d_ji_1, d_kj_1).norm(dim=1)

    angle = torch.atan2(b, a)
    return angle

class ZeoData(Data):
 
    def __inc__(self, key, value, *args, **kwargs):
       
        if key in ['idx_kj_1', 'idx_ji_1', 'edge_index_triplets']:
            return len(self.edge_attr)
        elif 'index' in key or 'idx' in key:
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
class CondData(Data):
 
    def __inc__(self, key, value, *args, **kwargs):
       
        if key == 'edge_index_ad':
            return len(self.x_ad)
        elif key == 'batch_ad':
            return 1
        elif 'index' in key or 'idx' in key:
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
def unpack_batch(batch, device : str = 'cpu'):
    '''
    Unpacks batch and moves data to device
 
    Parameters
    ----------
    batch : Batch
        batch of graph data
    device : str
        device to which data should be moved
 
    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        x, edge_index, edge_attr, orbit, orbit_weight, orbit_index, edge_orbit, edge_orbit_weight, edge_orbit_index, y, batch
    '''
    batch = batch.to(DEVICE)
    x = batch.x.long()
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr.float()
    y = batch.y.float()
    batch = batch.batch
   
    return (x, edge_index, edge_attr, y, batch)
        
def unpack_batch_ad(batch, device : str = 'cpu'):
    '''
    Unpacks batch and moves data to device
 
    Parameters
    ----------
    batch : Batch
        batch of graph data
    device : str
        device to which data should be moved
 
    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        x, edge_index, edge_attr, orbit, orbit_weight, orbit_index, edge_orbit, edge_orbit_weight, edge_orbit_index, y, batch
    '''
    batch = batch.to(DEVICE)
    x = batch.x.long()
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr.float()
    y = batch.y.float()
    btch = batch.batch
    x_ad = batch.x_ad
    edge_index_ad = batch.edge_index_ad
    edge_attr_ad = batch.edge_attr_ad.float()
    btch_ad = batch.batch_ad
    label = batch.label
 
    return (x, edge_index, edge_attr, y, btch, x_ad, edge_index_ad, edge_attr_ad, btch_ad, label)

def create_graphs(zeo : str = 'MOR', triplets : bool = False):
    """
    Creates list of graphs for the given zeolite structure
 
    zeo: str (default: 'MOR')
        Name of the zeolite structure
    triplets: bool (default: False)
        If True, returns graphs with triplets
    """
 
    graphs = []
 
    X = get_tensor(f'Data/{zeo}/X.npy')
    atoms = get_tensor(f'Data/{zeo}/atoms.npy')
    adj = get_tensor(f'Data/{zeo}/adj.npy')
    l = get_tensor(f'Data/{zeo}/l.npy')
    angles = get_tensor(f'Data/{zeo}/angles.npy')
    y = get_tensor(f'Data/{zeo}/hoa.npy')
 
    z_co2 = torch.tensor([8, 6, 8])
    x_co2 = torch.tensor([[0.0, 0.0, 1.149],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.149]])
    edge_index_co2 = torch.tensor([[0, 1, 0], [1, 2, 2]])
    edge_attr_co2 = (x_co2[edge_index_co2[0]] - x_co2[edge_index_co2[1]]).norm(dim=1).unsqueeze(1)

    co2 = [z_co2, x_co2, edge_index_co2, edge_attr_co2]

    z_ch4 = torch.tensor([6, 1, 1, 1, 1])
    x_ch4 = torch.tensor([[ 0.9993, -0.0025, -0.0044],
                        [ 2.0923, -0.0024,  0.0041],
                        [ 0.6344,  1.0279,  0.0041],
                        [ 0.6277, -0.5283,  0.8790],
                        [ 0.6420, -0.5080, -0.9063]])
    edge_index_ch4 = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                                    [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])
    edge_attr_ch4 = (x_ch4[edge_index_ch4[0]] - x_ch4[edge_index_ch4[1]]).norm(dim=1).unsqueeze(1)

    ch4 = [z_ch4, x_ch4, edge_index_ch4, edge_attr_ch4]

    z_h2 = torch.tensor([1, 1])
    x_h2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    edge_index_h2 = torch.tensor([[0, 1], [1, 0]])
    edge_attr_h2 = (x_h2[edge_index_h2[0]] - x_h2[edge_index_h2[1]]).norm(dim=1).unsqueeze(1)

    h2 = [z_h2, x_h2, edge_index_h2, edge_attr_h2]

    z_c2h6 = torch.tensor(2*[6]+6*[1])
    x_c2h6 = torch.tensor([[ 1.8974e+00, -5.5747e-03, -1.2094e-02],
            [ 4.7545e+00, -1.3266e-02,  1.1149e-03],
            [ 1.1761e+00,  1.9321e+00, -3.0992e-03],
            [ 1.1605e+00, -9.8217e-01,  1.6547e+00],
            [ 1.1761e+00, -9.6077e-01, -1.6981e+00],
            [ 5.4758e+00,  9.4883e-01,  1.6830e+00],
            [ 5.4913e+00,  9.5637e-01, -1.6697e+00],
            [ 5.4757e+00, -1.9510e+00,  1.1338e-04],])
                            
    num_nodes = 8
    adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
    edge_index_c2h6 = torch.stack(torch.where(adj_matrix == 1))
    edge_attr_c2h6 = (x_c2h6[edge_index_c2h6[0]] - x_c2h6[edge_index_c2h6[1]]).norm(dim=1).unsqueeze(1)

    c2h6 = [z_c2h6, x_c2h6, edge_index_c2h6, edge_attr_c2h6]
    
    z_c3h8 = torch.tensor(3*[6]+8*[1])
    x_c3h8 = torch.tensor([[ 1.8708e+00, -2.5247e-02,  7.7857e-03],
            [ 4.7412e+00,  3.1199e-02,  1.5609e-02],
            [ 5.7479e+00,  2.7200e+00,  1.2227e-02],
            [ 1.1922e+00, -1.9793e+00,  9.6376e-04],
            [ 1.1144e+00,  9.2290e-01, -1.6681e+00],
            [ 1.1062e+00,  9.1376e-01,  1.6852e+00],
            [ 5.4552e+00, -9.7406e-01, -1.6475e+00],
            [ 5.4454e+00, -9.6652e-01,  1.6874e+00],
            [ 5.1238e+00,  3.7431e+00, -1.6742e+00],
            [ 7.8160e+00,  2.7113e+00,  2.8743e-02],
            [ 5.0966e+00,  3.7583e+00,  1.6790e+00],])
               
    num_nodes = 11
    adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
    edge_index_c3h8 = torch.stack(torch.where(adj_matrix == 1))
    edge_attr_c3h8 = (x_c3h8[edge_index_c3h8[0]] - x_c3h8[edge_index_c3h8[1]]).norm(dim=1).unsqueeze(1)

    c3h8 = [z_c3h8, x_c3h8, edge_index_c3h8, edge_attr_c3h8]
 
    h = torch.tensor(get_transform_matrix(*l, *angles))
 
    # edges, distances and angles always remain the same for a zeolite toplogy
    idx_i, idx_j = torch.where(adj)
    edge_index = torch.stack([idx_i, idx_j])
    dists = get_distance(X[idx_i], X[idx_j], h)
 
    for i in range(atoms.shape[0]):
        x = 0
        dict_mo = {'co2': co2, 'ch4': ch4, 'h2': h2, 'c2h6': c2h6, 'c3h8': c3h8}
        for k in dict_mo:
            at = dict_mo[k]
            data = CondData(x=atoms[i], edge_index=edge_index, edge_attr=dists.unsqueeze(1), y=y[i, x], x_ad=at[0], edge_index_ad=at[2], edge_attr_ad=at[3], batch_ad=torch.zeros(at[0].size(0), dtype=torch.long ), label=k)
            x += 1
            graphs.append(data)
 
    return graphs

class Scaler():

    def __init__(self, data):

        self.min = data.min()
        self.max = data.max()
        
    def scale(self, x):
        
        return (x - self.min)/ (self.max - self.min)
    
    def unscale(self, x):

        return (self.max - self.min) * x + self.min

@torch.no_grad()
def predict(dataloader, model):
 
    model.eval()
    preds = []
    trues = []
    for _, data in enumerate(dataloader):
 
        x, edge_index, edge_attr, y, batch = unpack_batch(data, 'cuda')
        out = model(x, edge_index, edge_attr, batch).squeeze()
        preds.append(out.detach().cpu())
        trues.append(y.detach().cpu())
    return np.concatenate(preds), np.concatenate(trues)

@torch.no_grad()
def predict_ad(dataloader, model):
 
    model.eval()
    preds = []
    trues = []
    for _, data in enumerate(dataloader):
 
        x, edge_index, edge_attr, y, btch, x_ad, edge_index_ad, edge_attr_ad, btch_ad, label = unpack_batch_ad(data, 'cuda')
        out = model(x, edge_index, edge_attr, btch, x_ad, edge_index_ad, edge_attr_ad, btch_ad, label).squeeze()
        if len(out.shape) == 0:
            out = out[None]
        preds.append(out.detach().cpu())
        trues.append(y.detach().cpu())
    return np.concatenate(preds), np.concatenate(trues)

if __name__ == "__main__": #if file called, code below is executed
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', help='hidden_channels_and_nr_of_filters', type=int, default=128)
    parser.add_argument('--i', help='nr_of_interactions', type=int, default=6)
    parser.add_argument('--n', help='epochs', type=int, default=100)
    parser.add_argument('--requires_grad', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--m', help='model', type=str, default='SchNet')
    parser.add_argument('--x', help='x-parameter SetModel', type=int, default=128)
    parser.add_argument('--y', help='y-parameter SetModel', type=int, default=128)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mo', help='molecule', type=str, default=None)
    #more inputs can be added
    #possibly also the model
    args = parser.parse_args()

    NUM_ATOMS = 5
    if args.mo != None:
        NUM_ATOMS = NUM_ATOMS - 1

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    print(DEVICE)

    d1 = {'hidden_channels': 128, 'num_filters': 128, 'num_interactions': 6}

    if args.m == 'SchNet':  
        with open('Data/MOR/qm7_models/param_dict_schnet.pkl', 'rb') as f:
            d2 = pickle.load(f)
        atom_schnet = SchNet(d2['hidden_channels'], d2['num_filters'], d2['num_interactions'], num_gaussians=12).to(DEVICE)
        # d2 = {'hidden_channels': 4, 'num_filters': 4, 'num_interactions': 3}
        # atom_schnet = SchNet(d2['hidden_channels'], d2['num_filters'], d2['num_interactions'], num_gaussians=12).to(DEVICE)
        if args.pretrained:
            atom_schnet.load_state_dict(torch.load('Data/MOR/qm7_models/state_dict_schnet.pth', map_location=torch.device('cpu')))
        in_features_cond = atom_schnet.hidden_channels
    elif args.m == 'Set':
        with open('Data/MOR/qm7_models/param_dict_set.pkl', 'rb') as f:
            d2 = pickle.load(f)
        atom_schnet = SetModel(d2['x-value'], d2['y-value']).to(DEVICE)
        # d2 = {'x-value': 4, 'y-value': 4}
        # atom_schnet = SetModel(d2['x-value'], d2['y-value']).to(DEVICE)
        if args.pretrained:
            atom_schnet.load_state_dict(torch.load('Data/MOR/qm7_models/state_dict_set.pth', map_location=torch.device('cpu')))
        in_features_cond = atom_schnet.y
    elif args.m == 'Descriptor':
        atom_schnet = DescriptorModel().to(DEVICE)
        in_features_cond = 8

    zeo_schnet = SchNetZeo(in_features_cond = in_features_cond, cond_network = atom_schnet, **d1).to(DEVICE)

    # Freeze the parameters
    for param in zeo_schnet.cond_network.parameters():
        param.requires_grad = args.requires_grad

    h = torch.tensor(get_transform_matrix(18.256, 20.534, 7.5420, 90, 90, 90)).to(DEVICE)

    graphs = create_graphs('MOR', False)

    np.random.seed(1)

    if args.mo == None:
        test_size = 0.2
        n_train_gr = int((1 - test_size)*(len(graphs)//NUM_ATOMS))
        indices = np.arange(len(graphs)//NUM_ATOMS)

        np.random.shuffle(indices)
        indices_train = indices[:n_train_gr]
        indices_test = indices[n_train_gr:]

        indices_train = np.repeat(indices_train, NUM_ATOMS)

        for i in range(len(indices_train)):
            indices_train[i] = NUM_ATOMS*indices_train[i] + i%NUM_ATOMS

        indices_test = np.repeat(indices_test, NUM_ATOMS)

        for i in range(len(indices_test)):
            indices_test[i] = NUM_ATOMS*indices_test[i] + i%NUM_ATOMS
    else:
        indices_train = [i for i in range(len(graphs)) if graphs[i].label != args.mo]
        indices_test = [i for i in range(len(graphs)) if graphs[i].label == args.mo]

    GRAPHS_train = []
    GRAPHS_test = []

    for idx in indices_train:
        GRAPHS_train.append(graphs[idx])

    for idx in indices_test:
        GRAPHS_test.append(graphs[idx])

    # GRAPHS_train = graphs[indices_train.astype(int)]
    # GRAPHS_test = graphs[indices_test.astype(int)]

    graphs_train = DataLoader(GRAPHS_train, batch_size=32, shuffle=True)
    graphs_test = DataLoader(GRAPHS_test, batch_size=32, shuffle=False) # change batch_size

    print(graphs_train)
    print(graphs_test)

    epochs = args.n

    optimizer = torch.optim.AdamW(zeo_schnet.parameters(),) # change learning rate
    criterion = nn.L1Loss(reduction='none')

    tr_loss_gr = []
    te_loss_gr = []

    for epoch in tqdm(range(epochs)):

        zeo_schnet.train()
        running_loss = 0.0
        for i, data in enumerate(graphs_train):

            optimizer.zero_grad()

            x, edge_index, edge_attr, y, btch, x_ad, edge_index_ad, edge_attr_ad, btch_ad, label = unpack_batch_ad(data, DEVICE)
            out = zeo_schnet(x.long(), edge_index, edge_attr, btch, x_ad.long(), edge_index_ad, edge_attr_ad, btch_ad, label).squeeze()
            loss = criterion(out, y)/y
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # tr_loss_gr.append(loss.item())

        tr_loss_gr.append(running_loss/(i+1))
        # print(f'Epoch {epoch+1} loss: {running_loss/(i+1)}')

        zeo_schnet.eval()
        running_loss_test = 0.0
        for i, data in enumerate(graphs_test):

            x, edge_index, edge_attr, y, btch, x_ad, edge_index_ad, edge_attr_ad, btch_ad, label = unpack_batch_ad(data, DEVICE)
            with torch.no_grad():
                out = zeo_schnet(x.long(), edge_index, edge_attr, btch, x_ad.long(), edge_index_ad, edge_attr_ad, btch_ad, label).squeeze()
            loss = criterion(out, y)/y
            loss = loss.mean()

            running_loss_test += loss.item()
            # te_loss_gr.append(loss.item())

        te_loss_gr.append(running_loss_test/(i+1))
        # print(f'Epoch {epoch+1} test loss: {running_loss_test/(i+1)}')

    current_dir = os.getcwd()
    existing_folders = os.listdir(f'{current_dir}/saved_results/')
    #existing_folders = [int(i) for i in existing_folders]

    if len(existing_folders) == 1:
        next_dir = 0
    else:
        next_dir = len(existing_folders) - 1

    os.makedirs(f'{current_dir}/saved_results/{next_dir}/')
    torch.save(zeo_schnet.state_dict(), f'{current_dir}/saved_results/{next_dir}/state_dict.pth')
    torch.save(tr_loss_gr, f'{current_dir}/saved_results/{next_dir}/tr_loss_gr.py')
    torch.save(te_loss_gr, f'{current_dir}/saved_results/{next_dir}/te_loss_gr.py')
    if args.m != 'Descriptor':
        with open(f'{current_dir}/saved_results/{next_dir}/param_dict.pkl', 'wb') as f:
            pickle.dump(d2, f)