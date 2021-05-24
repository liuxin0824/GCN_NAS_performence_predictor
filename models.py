import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.fc1 = nn.Linear(16*13*nclass, 60) #16*13*nclass 32 后改为60
        self.fc2 = nn.Linear(60, 1) #32,1

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.fc1 = nn.Linear(208 * nclass, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))

        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)


class GCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.fc1 = nn.Linear(16 * 13 * nclass, 1)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)


        x = x.flatten()
        x = F.relu(self.fc1(x))


        return F.sigmoid(x)