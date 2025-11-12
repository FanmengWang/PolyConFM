import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import polyconfm.models.design_module.utils as utils
from polyconfm.models.design_module.diffusion import diffusion_utils
from polyconfm.models.design_module.metrics.train_loss import TrainLossDiscrete
from polyconfm.models.design_module.models.transformer import Denoiser
from polyconfm.models.design_module.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from polyconfm.models.design_module.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalTransition


class Graph_DiT(nn.Module):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.model.diffusion_steps
        self.test_only = cfg.general.test_only
        self.guide_scale = cfg.model.guide_scale

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist
        active_index = dataset_infos.active_index
        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist
        self.active_index = active_index
        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_collection = []

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_collection = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.max_n_nodes = dataset_infos.max_n_nodes

        self.model = Denoiser(max_n_nodes=self.max_n_nodes,
                        hidden_size=cfg.model.hidden_size,
                        depth=cfg.model.depth,
                        num_heads=cfg.model.num_heads,
                        mlp_ratio=cfg.model.mlp_ratio,
                        drop_condition=cfg.model.drop_condition,
                        Xdim=self.Xdim, 
                        Edim=self.Edim,
                        ydim=self.ydim,
                        task_type=dataset_infos.task_type)
        
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        x_marginals = self.dataset_info.node_types.float() / torch.sum(self.dataset_info.node_types.float())
        
        e_marginals = self.dataset_info.edge_types.float() / torch.sum(self.dataset_info.edge_types.float())
        x_marginals = x_marginals / (x_marginals ).sum()
        e_marginals = e_marginals / (e_marginals ).sum()

        xe_conditions = self.dataset_info.transition_E.float()
        xe_conditions = xe_conditions[self.active_index][:, self.active_index] 
        
        xe_conditions = xe_conditions.sum(dim=1) 
        ex_conditions = xe_conditions.t()
        xe_conditions = xe_conditions / xe_conditions.sum(dim=-1, keepdim=True)
        ex_conditions = ex_conditions / ex_conditions.sum(dim=-1, keepdim=True)
        
        self.transition_model = MarginalTransition(x_marginals=x_marginals, 
                                                          e_marginals=e_marginals, 
                                                          xe_conditions=xe_conditions,
                                                          ex_conditions=ex_conditions,
                                                          y_classes=self.ydim_output,
                                                          n_nodes=self.max_n_nodes)

        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals, y=None)

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps

        self.best_val_nll = 1e8
        self.val_counter = 0
        self.batch_size = self.cfg.train.batch_size
   
    def forward(self, noisy_data, whole_condition, unconditioned=False):
        x, e, y = noisy_data['X_t'].float(), noisy_data['E_t'].float(), noisy_data['y_t'].float().clone()
        node_mask, t = noisy_data['node_mask'], noisy_data['t']
        pred = self.model(x, e, node_mask, y=y, t=t, whole_condition=whole_condition, unconditioned=unconditioned)
        return pred
    
    def apply_noise(self, X, E, y, node_mask):
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = (t_int / self.T).to(self.device)
        s_float = (s_int / self.T).to(self.device)

        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        
        bs, n, d = X.shape
        X_all = torch.cat([X, E.reshape(bs, n, -1)], dim=-1).to(self.device)
        prob_all = X_all @ Qtb.X
        probX = prob_all[:, :, :self.Xdim_output]
        probE = prob_all[:, :, self.Xdim_output:].reshape(bs, n, n, -1)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).to(self.device)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).to(self.device)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        y_t = y.to(self.device)
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y_t).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data
    
    def training_step(self, data, whole_condition, device):
        self.device = device
        data_x = F.one_hot(data.x, num_classes=118).float()[:, self.active_index]
        data_edge_attr = F.one_hot(data.edge_attr, num_classes=5).float()
        dense_data, node_mask = utils.to_dense(data_x, data.edge_index, data_edge_attr, data.batch, self.max_n_nodes)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X.to(self.device), dense_data.E.to(self.device)
        node_mask = node_mask.to(self.device)
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        pred = self.forward(noisy_data, whole_condition)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                            true_X=X, true_E=E, true_y=data.y.to(self.device), node_mask=node_mask)
        return loss
    
    def sampling_step(self, data, whole_condition, device, sample_batch=125):
        self.device = device
        batch_y = data.y.to(self.device).repeat(sample_batch, 1)
        batch_whole_condition = whole_condition.repeat(sample_batch, 1)
        sample = self.sample_batch(batch_y.shape[0], batch_y, batch_whole_condition)
        return sample, batch_y
    
    def sample_batch(self, batch_size, y, whole_condition, num_nodes=None):
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = self.max_n_nodes
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E = z_T.X, z_T.E

        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, whole_condition, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        
        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, whole_condition, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        
        def get_prob(noisy_data, whole_condition, unconditioned=False):
            pred = self.forward(noisy_data, whole_condition, unconditioned=unconditioned)

            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

            # Retrieve transitions matrix
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
            Qt = self.transition_model.get_Qt(beta_t, self.device)

            Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
            p_s_and_t_given_0 = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=Xt_all,
                                                                                            Qt=Qt.X,
                                                                                            Qsb=Qsb.X,
                                                                                            Qtb=Qtb.X)
            predX_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)
            weightedX_all = predX_all.unsqueeze(-1) * p_s_and_t_given_0
            unnormalized_probX_all = weightedX_all.sum(dim=2)                     # bs, n, d_t-1

            unnormalized_prob_X = unnormalized_probX_all[:, :, :self.Xdim_output]
            unnormalized_prob_E = unnormalized_probX_all[:, :, self.Xdim_output:].reshape(bs, n*n, -1)

            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5

            prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1
            prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)  # bs, n, d_t-1
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

            return prob_X, prob_E

        prob_X, prob_E = get_prob(noisy_data, whole_condition)

        ### Guidance
        if self.guide_scale is not None and self.guide_scale != 1:
            uncon_prob_X, uncon_prob_E = get_prob(noisy_data, whole_condition, unconditioned=True)
            prob_X = uncon_prob_X *  (prob_X / uncon_prob_X.clamp_min(1e-10)) ** self.guide_scale  
            prob_E = uncon_prob_E * (prob_E / uncon_prob_E.clamp_min(1e-10)) ** self.guide_scale  
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-10)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask, step=s[0,0].item())

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)