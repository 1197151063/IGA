import world
from re_model import IGCL
from re_dataloader import Loader
import torch
from re_procedure import test,train_bpr_simgcl
import utils
device = world.device
config = world.config
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = IGCL(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.

for epoch in range(1, 1001):
    loss = train_bpr_simgcl(dataset=dataset,
                         model=model,
                         opt=opt)
    recall,ndcg = test([20],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f}'
          f', N@20: {ndcg[20]:.4f}')