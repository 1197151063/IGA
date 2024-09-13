from torch import nn,Tensor,LongTensor
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
import world
import utils
config = world.config
device = world.device
"""
define Recmodels here:
Already implemented : IGA
"""
"""
General GNN based RecModel
"""
class RecModel(MessagePassing):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.config = config
        self.f = nn.Sigmoid()


    def get_sparse_graph(self,
                         edge_index,
                         use_value=False,
                         value=None):
        num_users = self.num_users
        num_nodes = self.num_nodes
        r,c = edge_index
        row = torch.cat([r , c + num_users])
        col = torch.cat([c + num_users , r])
        if use_value:
            value = torch.cat([value,value])
            return SparseTensor(row=row,col=col,value=value,sparse_sizes=(num_nodes,num_nodes))
        else:
            return SparseTensor(row=row,col=col,sparse_sizes=(num_nodes,num_nodes))
    
    def get_embedding(self):
        raise NotImplementedError
    
    def forward(self,
                edge_label_index:Tensor):
        out = self.get_embedding()
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        out_src = out_u[edge_label_index[0]]
        out_dst = out_i[edge_label_index[1]]
        out_dst_neg = out_i[edge_label_index[2]]
        return (out_src * out_dst).sum(dim=-1),(out_src * out_dst_neg).sum(dim=-1)
    
    def link_prediction(self,
                        src_index:Tensor=None,
                        dst_index:Tensor=None):
        out = self.get_embedding()
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        if src_index is None:
            src_index = torch.arange(self.num_users).long()
        if dst_index is None:
            dst_index = torch.arange(self.num_items).long()
        out_src = out_u[src_index]
        out_dst = out_i[dst_index]
        pred = out_src @ out_dst.t()
        return pred
    
    def recommendation_loss(self,
                            pos_rank,
                            neg_rank,
                            edge_label_index):
        rec_loss = torch.nn.functional.softplus(neg_rank - pos_rank).mean()
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[edge_label_index[2]]])
        regularization = self.config['decay'] * (1/2) * embedding.norm(p=2).pow(2)
        regularization = regularization / pos_rank.size(0)
        return rec_loss , regularization
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)
    

class IGCL(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(num_users=num_users,
                         num_items=num_items,
                         edge_index=edge_index,
                         config=config
                         )
        self.K = config['K']
        self.num_interactions = edge_index.size(1)
        self.edge_index = self.get_sparse_graph(edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(self.edge_index)        
        self.alpha= 1./ (self.K)
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['latent_dim_rec'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['latent_dim_rec'])
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K))
        self.ssl_tmp = config['ssl_tmp']
        self.ssl_decay = config['ssl_decay']
        self.eps = config['epsilon']
        print('Go backbone SimGCL')
        print(f"params settings: \n L2 reg:{config['decay']}\n layer:{self.K}")
        print(f" ssl_tmp:{config['ssl_tmp']}\n ssl_decay:{config['ssl_decay']}\n noise_bias:{config['epsilon']}")
    
    def norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()).sqrt()
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()).sqrt()
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
    
        return x
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = []
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out.append(x)
        out = torch.stack(out,dim=1)
        out = torch.mean(out,dim=1)
        return out
    
    def get_shuffle_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = []
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            r_noise = torch.rand_like(x).cuda()
            x = x + torch.sign(x) * F.normalize(r_noise,dim=-1) * self.eps
            out.append(x)
        out = torch.stack(out,dim=1)
        out = torch.mean(out,dim=1)
        return out
    
    def ssl_loss(self,edge_label_index):
        u_idx,i_idx = edge_label_index[0],edge_label_index[1]
        view1 = self.get_shuffle_embedding()
        view2 = self.get_shuffle_embedding()
        info_out_u_1,info_out_i_1 = torch.split(view1,[self.num_users,self.num_items])
        info_out_u_2,info_out_i_2 = torch.split(view2,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        user_cl_loss = utils.InfoNCE(info_out_u_1[u_idx], info_out_u_2[u_idx], 0.2)
        item_cl_loss = utils.InfoNCE(info_out_i_1[i_idx], info_out_i_2[i_idx], 0.2)
        return self.ssl_decay * (user_cl_loss + item_cl_loss)    
    
    def focal_ssl_loss(self,
                       edge_label_index):
        info_out1 = self.get_shuffle_embedding()
        info_out2 = self.get_shuffle_embedding()
        info_out_u_1,info_out_i_1 = torch.split(info_out1,[self.num_users,self.num_items])
        info_out_u_2,info_out_i_2 = torch.split(info_out2,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        info_out_u1 = info_out_u_1[u_idx]
        info_out_u2 = info_out_u_2[u_idx]
        info_out_i1 = info_out_i_1[i_idx]
        info_out_i2 = info_out_i_2[i_idx]
        info_out_u1 = F.normalize(info_out_u1,dim=1)
        info_out_u2 = F.normalize(info_out_u2,dim=1)
        info_pos_user = (info_out_u1 * info_out_u2).sum(dim=1)/ self.ssl_tmp
        info_pos_user = torch.exp(info_pos_user)
        info_neg_user = (info_out_u1 @ info_out_u2.t())/ self.ssl_tmp
        info_neg_user = torch.exp(info_neg_user)
        info_neg_user = torch.sum(info_neg_user,dim=1,keepdim=True)
        ssl_logits_user = -torch.log(info_pos_user / info_neg_user).mean()
        info_out_i1 = F.normalize(info_out_i1,dim=1)
        info_out_i2 = F.normalize(info_out_i2,dim=1)
        info_pos_item = (info_out_i1 * info_out_i2).sum(dim=1)/ self.ssl_tmp
        info_neg_item = (info_out_i1 @ info_out_i2.t())/ self.ssl_tmp
        info_pos_item = torch.exp(info_pos_item)
        info_neg_item = torch.exp(info_neg_item)
        info_neg_item = torch.sum(info_neg_item,dim=1,keepdim=True)
        ssl_logits_item = -torch.log(info_pos_item / info_neg_item).mean()
        return self.ssl_decay * (ssl_logits_user + ssl_logits_item) 

    def bpr_loss(self,pos_rank,neg_rank):
        return F.softplus(neg_rank - pos_rank).mean()
    
    def L2_reg(self,edge_label_index):
        u_idx,i_idx_pos,i_idx_neg = edge_label_index
        userEmb0 = self.user_emb.weight[u_idx]
        posEmb0 = self.item_emb.weight[i_idx_pos]
        negEmb0 = self.item_emb.weight[i_idx_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / edge_label_index.size(1)
        regularization = self.config['decay'] * reg_loss
        return regularization
    