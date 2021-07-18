# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:33:01 2020

@author: Liwei Huang
"""
import torch
import torch.nn as nn
from modules import TemporalAttentionLayer

class PTGCN(nn.Module):
    '''
    Position enhanced and Time aware graph convolution
    '''
    def __init__(self, user_neig50, item_neig50, num_users, num_items, time_encoder, n_layers, n_neighbors,
               n_node_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1):
        super(PTGCN, self).__init__()
    
        self.num_users = num_users
        self.num_items = num_items
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.n_neighbors = n_neighbors
    
        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_dimension)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_dimension)
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)
        
        self.user_neig50, self.user_egdes50, self.user_neig_time50, self.user_neig_mask50 = user_neig50
        self.item_neig50, self.item_egdes50, self.item_neig_time50, self.item_neig_mask50 = item_neig50
    
        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            time_dim=n_time_features,
            output_dimension=embedding_dimension,   
            n_head=n_heads,
            n_neighbor= self.n_neighbors[i],
            dropout=dropout)
            for i in range(n_layers)])

    def compute_embedding(self, nodes, edges, timestamps, n_layers, nodetype='user'):
        """
        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """
        assert (n_layers >= 0)
        n_neighbor = self.n_neighbors[n_layers-1]
        nodes_torch = nodes.long()
        edges_torch = edges.long()
        timestamps_torch = timestamps.long()

        # query node always has the start time -> time span == 0
        nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))
        if nodetype=='user':
            node_features = self.user_embeddings(nodes_torch)
        else:
            node_features = self.item_embeddings(nodes_torch)
            
        if n_layers == 0:
            return node_features
        else:
            if nodetype=='user':
                
                adj, adge, times, mask = self.user_neig50[edges_torch,-n_neighbor:], self.user_egdes50[edges_torch,-n_neighbor:], self.user_neig_time50[edges_torch,-n_neighbor:], self.user_neig_mask50[edges_torch,-n_neighbor:]
                
                edge_deltas = timestamps_torch.unsqueeze(1) - times   #[batch_size,n_neighors]
                adj = adj.flatten()
                times = times.flatten()
                adge = adge.flatten()

                neighbor_embeddings = self.compute_embedding(adj, adge, times, n_layers - 1, 'item')
                neighbor_embeddings = neighbor_embeddings.view(len(nodes), n_neighbor, -1)
                edge_time_embeddings = self.time_encoder(edge_deltas.flatten())
                edge_time_embeddings = edge_time_embeddings.view(len(nodes), n_neighbor, -1)

                node_embedding,_  = self.attention_models[n_layers - 1](node_features,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        mask)
            
        
            if nodetype=='item':
                adj, adge, times, mask = self.item_neig50[edges_torch,-n_neighbor:], self.item_egdes50[edges_torch,-n_neighbor:], self.item_neig_time50[edges_torch,-n_neighbor:], self.item_neig_mask50[edges_torch,-n_neighbor:]
                
                edge_deltas = timestamps_torch.unsqueeze(1) - times   #[batch_size,n_neighors]
                adj = adj.flatten()
                times = times.flatten()
                adge = adge.flatten()
                
                neighbor_embeddings = self.compute_embedding(adj, adge, times, n_layers - 1, 'user')
                neighbor_embeddings = neighbor_embeddings.view(len(nodes), n_neighbor, -1)
                edge_time_embeddings = self.time_encoder(edge_deltas.flatten())
                edge_time_embeddings = edge_time_embeddings.view(len(nodes), n_neighbor, -1)

                node_embedding,_ = self.attention_models[n_layers - 1](node_features,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        mask)
            
        
        return node_embedding
    
    
    def forward(self, nodes, edges, timestamps, n_layers, nodetype='user'):
        
        return self.compute_embedding(nodes, edges, timestamps, n_layers, nodetype)
        