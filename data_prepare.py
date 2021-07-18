# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:33:01 2020

@author: Liwei Huang
"""
import numpy as np
import pandas as pd
import torch
import os
import random

class NeighborFinder:
    def __init__(self, ratings):

        self.ratings = np.array(ratings)

        users = ratings['user_id'].unique()
        items = ratings['item_id'].unique()
        self.user_edgeidx = {cur_user: np.array(ratings[ratings.user_id == cur_user].index.tolist()) for cur_user in
                             users}  # 用户的边ID集合
        self.item_edgeidx = {cur_item: np.array(ratings[ratings.item_id == cur_item].index.tolist()) for cur_item in
                             items}  # item的边ID集合

    def get_user_neighbor(self, source_idx, timestamps, n_neighbors, device):

        assert (len(source_idx) == len(timestamps))

        adj_user = torch.zeros((len(source_idx), n_neighbors), dtype=torch.int32).to(device)  # 表示每一个节点的邻居向量
        user_mask = torch.ones((len(source_idx), n_neighbors), dtype=torch.bool).to(device)
        user_time = torch.zeros((len(source_idx), n_neighbors), dtype=torch.int32).to(
            device)  # time matirx，节点与其他max_nodes的时间差
        adj_user_edge = torch.zeros((len(source_idx), n_neighbors), dtype=torch.int32).to(device)

        edge_idxs = torch.searchsorted(self.ratings[:, 2], timestamps)

        for i in range(len(source_idx)):
            idx = torch.searchsorted(self.user_edgeidx[source_idx[i].item()], edge_idxs[i].item())  # 当前用户最近的边
            his_len = len(self.user_edgeidx[source_idx[i].item()][:idx])
            used_len = his_len if his_len <= n_neighbors else n_neighbors

            adj_user[i, n_neighbors - used_len:] = self.ratings[:, 1][
                self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]]
            user_time[i, n_neighbors - used_len:] = self.ratings[:, 2][
                self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]]
            user_mask[i, n_neighbors - used_len:] = 0
            adj_user_edge[i, n_neighbors - used_len:] = self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]

        return adj_user, adj_user_edge, user_time, user_mask

    def get_item_neighbor(self, destination_idx, timestamps, n_neighbors, device):

        assert (len(destination_idx) == len(timestamps))

        adj_item = torch.zeros((len(destination_idx), n_neighbors), dtype=torch.int32).to(device)  # 表示每一个节点的邻居向量
        item_mask = torch.ones((len(destination_idx), n_neighbors), dtype=torch.bool).to(device)
        item_time = torch.zeros((len(destination_idx), n_neighbors), dtype=torch.int32).to(device)
        adj_item_edge = torch.zeros((len(destination_idx), n_neighbors), dtype=torch.int32).to(device)

        edge_idxs = torch.searchsorted(self.ratings[:, 2], timestamps)

        for i in range(len(destination_idx)):
            idx = torch.searchsorted(self.item_edgeidx[destination_idx[i].item()], edge_idxs[i].item())  # 当前用户最近的边
            his_len = len(self.item_edgeidx[destination_idx[i].item()][:idx])
            used_len = his_len if his_len <= n_neighbors else n_neighbors

            adj_item[i, n_neighbors - used_len:] = self.ratings[:, 0][
                self.item_edgeidx[destination_idx[i].item()][idx - used_len:idx]]
            item_time[i, n_neighbors - used_len:] = self.ratings[:, 2][
                self.item_edgeidx[destination_idx[i].item()][idx - used_len:idx]]
            item_mask[i, n_neighbors - used_len:] = 0
            adj_item_edge[i, n_neighbors - used_len:] = self.item_edgeidx[destination_idx[i].item()][idx - used_len:idx]

        return adj_item, adj_item_edge, item_time, item_mask

    def get_user_neighbor_ind(self, source_idx, edge_idx, n_neighbors, device):

        adj_user = np.zeros((len(edge_idx), n_neighbors), dtype=np.int32)  # 表示每一个节点的邻居向量
        user_mask = np.ones((len(edge_idx), n_neighbors), dtype=np.bool)
        user_time = np.zeros((len(edge_idx), n_neighbors), dtype=np.int32)  # time matirx，节点与其他max_nodes的时间差
        adj_user_edge = np.zeros((len(source_idx), n_neighbors), dtype=np.int32)

        for i in range(len(edge_idx)):
            idx = np.searchsorted(self.user_edgeidx[source_idx[i]], edge_idx[i]) + 1
            his_len = len(self.user_edgeidx[source_idx[i]][:idx])
            used_len = his_len if his_len <= n_neighbors else n_neighbors

            adj_user[i, n_neighbors - used_len:] = self.ratings[:,1][self.user_edgeidx[source_idx[i]][idx - used_len:idx]]
            user_time[i, n_neighbors - used_len:] = self.ratings[:,2][self.user_edgeidx[source_idx[i]][idx - used_len:idx]]
            user_mask[i, n_neighbors - used_len:] = 0
            adj_user_edge[i, n_neighbors - used_len:] = self.user_edgeidx[source_idx[i]][idx - used_len:idx]

        return torch.from_numpy(adj_user).to(device), torch.from_numpy(adj_user_edge).to(device), torch.from_numpy(
            user_time).to(device), torch.from_numpy(user_mask).to(device)

    def get_item_neighbor_ind(self, destination_idx, edge_idx, n_neighbors, device):

        adj_item = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)  # 表示每一个节点的邻居向量
        item_mask = np.ones((len(destination_idx), n_neighbors), dtype=np.bool)
        item_time = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)
        adj_item_edge = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)

        for i in range(len(destination_idx)):
            idx = np.searchsorted(self.item_edgeidx[destination_idx[i]], edge_idx[i]) + 1
            his_len = len(self.item_edgeidx[destination_idx[i]][:idx])
            used_len = his_len if his_len <= n_neighbors else n_neighbors

            adj_item[i, n_neighbors - used_len:] = self.ratings[:,0][self.item_edgeidx[destination_idx[i]][idx - used_len:idx]]
            item_time[i, n_neighbors - used_len:] = self.ratings[:,2][self.item_edgeidx[destination_idx[i]][idx - used_len:idx]]
            item_mask[i, n_neighbors - used_len:] = 0
            adj_item_edge[i, n_neighbors - used_len:] = self.item_edgeidx[destination_idx[i]][idx - used_len:idx]

        return torch.from_numpy(adj_item).to(device), torch.from_numpy(adj_item_edge).to(device), torch.from_numpy(
            item_time).to(device), torch.from_numpy(item_mask).to(device)


def data_partition(fname):
    
    # read ratings
    ratings = []
    with open(os.path.join(fname, 'ratings.dat')) as f:
        for l in f:
            user_id, item_id, rating, timestamp = [int(_) for _ in l.split('::')]
            ratings.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
    ratings = pd.DataFrame(ratings)

    ratings['timestamp'] = ratings['timestamp'] - min(ratings['timestamp'])   

    users = ratings['user_id'].unique()
    items = ratings['item_id'].unique()

    user_ids_invmap = {id_: i for i, id_ in enumerate(users)}
    item_ids_invmap = {id_: i for i, id_ in enumerate(items)}
    ratings['user_id'].replace(user_ids_invmap, inplace=True)
    ratings['item_id'].replace(item_ids_invmap, inplace=True)

    print('user_count:'+str(len(users))+','+'item_count:'+str(len(items)))
    print('avr of user:'+str(ratings['user_id'].value_counts().mean())+'avr of item:'+str(ratings['item_id'].value_counts().mean()))
    print(len(ratings))

    users = ratings['user_id'].unique()
    items = ratings['item_id'].unique()
    
    ratings = ratings.sort_values(by='timestamp',ascending=True)  
    ratings = ratings.reset_index(drop=True)
    full_data = []
    
    adj_user = {cur_user:ratings[ratings.user_id == cur_user].index.tolist() for cur_user in users} 
    adj_item = {cur_item:ratings[ratings.item_id == cur_item].index.tolist() for cur_item in items}
    
    for i in range(ratings.shape[0]):  #edge ID
        
        cur_user = ratings['user_id'].iloc[i]
        cur_item = ratings['item_id'].iloc[i]
        #确保训练集和测试集中的序列至少含有3个邻居
        if adj_user[cur_user].index(i)>=3 and adj_item[cur_item].index(i)>=3:
            full_data.append(i)
          
    offset1 = int(len(full_data) * 0.8)
    offset2 = int(len(full_data) * 0.9)
    random.shuffle(full_data)
    train_data, valid_data, test_data = full_data[0:offset1], full_data[offset1:offset2], full_data[offset2:len(full_data)]
   
    del ratings['rating']
    print(ratings.columns)
    
    return ratings, train_data, valid_data, test_data

if __name__ == '__main__':
    ratings, train_data, valid_data, test_data = data_partition('data/movielens/ml-1m')