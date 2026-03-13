import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()] 
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x): # B: cal_points
        # x: # [B, N, D]  
        batch_size = 120000 # CUDA OOM
        if x.shape[0] > batch_size:
            print('Using forward_batch', x.shape)
            A = []
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i+batch_size]
                a_batch = self.attention_a(x_batch) # [B1, N, D]
                b_batch = self.attention_b(x_batch) # [B1, N, D]
                A_batch = a_batch.mul(b_batch) # [B1, N, D]
                A_batch = self.attention_c(A_batch)
                A.append(A_batch)
                
            A = torch.cat(A, dim=0) # [B,N,D]
        else:
            a = self.attention_a(x) # tanh [B, N, 256]
            b = self.attention_b(x) # sigmoid [B, N, 256]
            A = a.mul(b) # [B, N, 256]
            A = self.attention_c(A)  # [B, N, 1]
        return A
     

class GeometryFeatureNet_weight_big(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=256, using_weight=True, temp=1, attn_fc=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, output_dim)
        )
        
        self.using_weight = using_weight

        attention_net = Attn_Net_Gated(L = output_dim, D = output_dim, dropout = False, n_classes = 1)
        if attn_fc:
            attention_net_all = [nn.Linear(output_dim, output_dim), nn.ReLU()]
            attention_net_all.append(attention_net)
            self.attention_net = nn.Sequential(*attention_net_all)
        else:
            self.attention_net = attention_net
        # self.attention_net = Attn_Net_Gated(L = output_dim, D = output_dim, dropout = False, n_classes = 1)
        

        self.temp = temp
        print('Model Temp: %s' % self.temp)



    def forward(self, ref_points, cal_points, return_A=False, L2_norm=True):
        # x: [B, N, 4] N: n_ref  
        # print(ref_points.shape, cal_points.shape)  [B, N, 2] [B, 1, 2]
        # print(ref_points, cal_points)

        delta = cal_points - ref_points  # [B, N, 2]
        x = torch.cat((delta, ref_points), dim=2)  # => [B, N, 4]  # x = torch.cat((delta, ref_points), dim=1) => [B, 2N, 2]

        x = self.mlp(x)                # [B, N, D]

        if not self.using_weight:
            x = x.mean(dim=1) # [B, D]
        else:
            
            A = self.attention_net(x)
            A = torch.transpose(A, 1, 2) # [B, 1, N] 
            A = F.softmax(A / self.temp, dim=2)  # softmax over N

            # [B, 1, N] @ [B, N, D] => [B, 1, D]
            weighted_sum = torch.bmm(A, x)  # [B, 1, D] [5120, 1, 128]
            
            x = weighted_sum.squeeze(1)  # [B, D] [5120, 128]
            # print(x.shape)
        
        if L2_norm:
            x = F.normalize(x, dim=1)
        
        if return_A:
            return x, A.detach().cpu()
        return x 


class GeometryFeatureNet_weight_big_split(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=256, using_weight=True, temp=1, attn_fc=False):
        super().__init__()

        self.delta_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.ref_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.together_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, output_dim)
        )
        
        self.using_weight = using_weight

        attention_net = Attn_Net_Gated(L = output_dim, D = output_dim, dropout = False, n_classes = 1)
        if attn_fc:
            attention_net_all = [nn.Linear(output_dim, output_dim), nn.ReLU()]
            attention_net_all.append(attention_net)
            self.attention_net = nn.Sequential(*attention_net_all)
        else:
            self.attention_net = attention_net
        
        self.temp = temp 
        print('Model Temp: %s' % self.temp)



    def forward(self, ref_points, cal_points, return_A=False, L2_norm=True): 
        # x: [B, N, 4] N: n_ref 
        # print(ref_points.shape, cal_points.shape)  [B, N, 2] [B, 1, 2]

        delta = cal_points - ref_points  # [B, N, 2]
       
        delta_fc = self.delta_mlp(delta) # [B, N, dim]
        ref_fc = self.ref_mlp(ref_points) # [B, N, dim]
        # print(delta_fc.shape, ref_fc.shape)
        x = torch.cat((delta_fc, ref_fc), dim=2) # [B, N, 2*dim]
        x = self.together_mlp(x)                # [B, N, D]

        if not self.using_weight:
            x = x.mean(dim=1) # [B, D]
        else:

            A = self.attention_net(x)
            A = torch.transpose(A, 1, 2) # [B, 1, N] 
            A = F.softmax(A / self.temp, dim=2)  # softmax over N

            # [B, 1, N] @ [B, N, D] => [B, 1, D]
            weighted_sum = torch.bmm(A, x)  # [B, 1, D] [5120, 1, 128]
            
            x = weighted_sum.squeeze(1)  # [B, D] [5120, 128]
            # print(x.shape)
        
        if L2_norm:
            x = F.normalize(x, dim=1)
        
        if return_A:
            return x, A.detach().cpu()
        return x   


