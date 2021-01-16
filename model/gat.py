import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, bias=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.bias = bias

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_list):
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.matmul(x, self.weight)

        source, target = edge_list
        a_input = torch.cat([h[source], h[target]], dim=1)
        e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=self.alpha)

        N = h.size(0)
        attention = -1e20 * torch.ones([N, N], device=DEVICE, requires_grad=True)
        attention[source, target] = e[:, 0]
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h = F.dropout(h, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.bias is not None:
            h_prime = h_prime + self.bias

        return h_prime


class GAT(nn.Module):
    def __init__(self, data, nhid=8, nhead=8, nhead_out=1, alpha=0.2, dropout=0.6):
        super(GAT, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nhead)]
        self.out_atts = [GraphAttention(nhid * nhead, nclass, dropout=dropout, alpha=alpha) for _ in range(nhead_out)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        for i, attention in enumerate(self.out_atts):
            self.add_module('out_att{}'.format(i), attention)
        self.reset_parameters()

    def reset_parameters(self):
        for att in self.attentions:
            att.reset_parameters()
        for att in self.out_atts:
            att.reset_parameters()

    def forward(self, data):
        x, edge_list = data.features, data.edge_list
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = F.elu(x)
        x = torch.sum(torch.stack([att(x, edge_list) for att in self.out_atts]), dim=0) / len(self.out_atts)
        return F.log_softmax(x, dim=1)
