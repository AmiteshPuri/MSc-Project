from torch.autograd import Function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import APPNPConv, GraphConv
import utils
from tqdm import tqdm
import numpy as np
import wandb
import copy
import pickle
import networkx as nx
import scipy.sparse as sp
import dgl
import os
from sklearn.metrics import f1_score
import models
from torch_geometric.logging import log

global task
task = 'classification'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'citeseer', 'DBLP_CF', 'pubmed', 'Amazon-Computers', 'Amazon-Photo',
                             'Coauthor-CS', 'Coauthor-Physics', 'Anaheim', 'ChicagoSketch', 'county_education_2012',
                             'county_election_2016', 'county_income_2012', 'county_unemployment_2012', 'twitch_PTBR'])
parser.add_argument('--hiddens', type=list, default=[32 * 4])
parser.add_argument('--k', type=int, default=12) # cora, citeseer k=12, pubmed = 8 in APPNP
parser.add_argument('--model', type=str, default='APPNP', choices=['APPNP', 'GAT', 'GCN', 'GraphSAGE', 'DAGNN'])
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--METHOD', type=str, default=None, choices=['SRGNN','cmd', 'MMD','SRKL','SRJS','kld','jsd','emd', None])
parser.add_argument('--TRAININGSET', type=str, default='Baised', choices=['Baised', None])
parser.add_argument('--BIASDATA', type=str, default='generate_own_bias_data', choices=['generate_own_bias_data',None])
parser.add_argument('--alpha_bias', type=float, default=0.1) # biasing parameter alpha(I-(1-alpha)\hat(A))^{-1}
parser.add_argument('--alpha_cov', type=float, default=0.1) # probabality relaxation (coverage is > 93%)
parser.add_argument('--alpha_cov1', type=float, default=0.05)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--epochs_base', type=int, default=100)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--device', type=str, default='cpu',choices=['cpu','gpu'])
parser.add_argument('--conformal_score', type=str, default='aps', choices=['aps', 'raps'])

parser.add_argument('--conftr', action='store_true', default=True)
parser.add_argument('--conftr_holdout', action='store_true', default=False)
parser.add_argument('--conftr_calib_holdout', action='store_true', default=True)

parser.add_argument('--bias_train', action='store_true', default=True)
###################################
parser.add_argument('--conftr_valid_holdout', action='store_true', default=False)
parser.add_argument('--srgnn', action='store_true', default=False)
# parser.add_argument('--conftr_valid_holdout', action='store_true', default=False)

parser.add_argument('--conftr_sep_test', action='store_true', default=False)
parser.add_argument('--conf_correct_model', type=str, default='gnn',
                    choices=['gnn', 'mlp', 'Calibrate', 'mcdropout', 'mcdropout_std', 'QR'])
parser.add_argument('--calibrator', type=str, default='NULL', choices=['TS', 'VS', 'ETS', 'CaGCN', 'GATS'])

parser.add_argument('--quantile', action='store_true', default=False)
parser.add_argument('--bnn', action='store_true', default=False)
###################################

parser.add_argument('--target_size', type=int, default=1)
parser.add_argument('--confgnn_hiddens', type=list, default=[16, 16])
parser.add_argument('--confgnn_num_layers', type=int, default=2)
# parser.add_argument('--confgnn_base_model', type=str, default='GCN', choices=['GAT', 'GCN', 'GraphSAGE', 'SGC'])
parser.add_argument('--confgnn_lr', type=float, default=1e-3)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--size_loss', action='store_true', default=False)
parser.add_argument('--size_loss_weight', type=float, default=1)
parser.add_argument('--reg_loss_weight', type=float, default=1)
parser.add_argument('--base_retrain', action='store_true', default=True)

parser.add_argument('--not_save_res', action='store_true', default=False)
parser.add_argument('--num_runs', type=int, default=1)

parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--hyperopt', action='store_true', default=False)
parser.add_argument('--optimal', action='store_true', default=True)
parser.add_argument('--optimal_examine', action='store_true', default=False)
parser.add_argument('--cond_cov_loss', action='store_true', default=False)
parser.add_argument('--conformal_training', action='store_true', default=False)

parser.add_argument('--ablation', type=str, default='NULL', choices=['NULL', 'mlp_conf_loss',
                                                                     'gnn_no_conf_loss',
                                                                     'Calibrate'
                                                                     ])

parser.add_argument('--calib_fraction', type=float, default=0.5)
parser.add_argument('--optimize_conformal_score', type=str, default='raps', choices=['aps', 'raps'])

args = parser.parse_args()


if args.TRAININGSET == 'Baised':
    name = 'baised_train_' + args.dataset + '_' + args.model + '_' + args.conformal_score #+ '_method_' + args.METHOD
    if args.METHOD == 'SRGNN':
        name += '_method_' + args.METHOD
    elif args.METHOD == 'cmd':
        name += '_method_' + args.METHOD  
    elif args.METHOD == 'MMD':
        name += '_method_' + args.METHOD
    elif args.METHOD == 'kld':
        name += '_method_' + args.METHOD
    elif args.METHOD == 'jsd':
        name += '_method_' + args.METHOD
    elif args.METHOD == 'SRKL':
        name += '_method_' + args.METHOD
    elif args.METHOD == 'SRJS':
        name += '_method_' + args.METHOD
    elif args.METHOD == 'emd':
        name += '_method_' + args.METHOD


    # if args.ablation != 'NULL':
    #     name += '_ablation_' + args.ablation
    if args.calib_fraction != 0.5:
        name += '_calib_fraction_' + str(args.calib_fraction)
    if args.alpha_bias != 0.1:
        name += 'alpha_bias_' + str(args.alpha_bias)
else:
    name = args.dataset + '_' + args.model + '_' + args.conformal_score + '_method_' + args.METHOD


if args.conf_correct_model == 'Calibrate':
    name += '_' + args.calibrator
elif args.conf_correct_model in ['mcdropout', 'QR', 'mcdropout_std']:
    name += '_' + args.conf_correct_model

if args.alpha_cov != 0.1:
    name += '_alpha_cov' + str(args.alpha_cov)

if args.size_loss:
    name += '_size_loss'
    if args.conftr:
        name += '_conftr'
    if args.conftr_calib_holdout:
        name += '_calib_holdout'
    if args.conf_correct_model == 'gnn':
        name += '_confgnn'
    if args.cond_cov_loss:
        name += '_cond_cov_loss'



# if args.optimize_conformal_score == 'raps':
    # name += '_raps'

if args.wandb:
    wandb.init(project='ConformalGNN_' + args.dataset + '_' + args.model, name=name, config=args)


def run_conformal_classification(pred, y, test_mask, n, alpha, score,
                                 calib_eval=True, validation_set=False,
                                 use_additional_calib=False, return_prediction_sets=False, calib_fraction=0.5):
    if calib_eval:
        n_base = int(n * (1 - calib_fraction))
    else:
        n_base = n

    logits = torch.nn.Softmax(dim=1)(pred).detach().cpu().numpy()
    # print(len(test_mask), pred.shape, n_base, n)

    smx = logits[test_mask]
    # labels = y[test_mask.expand_as(logits)].reshape(-1, logits.shape[1])
    labels = y[test_mask]
    # print(len(test_mask), labels.shape, smx.shape, logits.shape, n_base, n)

    cov_all = []
    eff_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []

    for k in range(100):
        idx = np.array([1] * n_base + [0] * (smx.shape[0] - n_base)) > 0
        # print(len(idx), smx.shape,labels.shape)
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_smx, val_smx = smx[idx, :], smx[~idx, :]
        cal_labels, val_labels = labels[idx], labels[~idx]

        # if use_additional_calib and calib_eval:
        #     smx_add = logits[data.calib_eval_mask]
        #     labels_add = data.y[data.calib_eval_mask].detach().cpu().numpy()
        #     cal_smx = np.concatenate((cal_smx, smx_add))
        #     cal_labels = np.concatenate((cal_labels, labels_add))

        n = cal_smx.shape[0]

        if score == 'aps':
            prediction_sets, cov, eff = utils.aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
        elif score == 'raps':
            prediction_sets, cov, eff = utils.raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)

        # prediction_sets, cov, eff = aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
        # print(prediction_sets, cov, eff)

        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)

    if return_prediction_sets:
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)

@torch.no_grad()
def test(model, x, edge_index, idx_test, valid_mask, y, alpha, tau, target_size, size_loss=False):
    model.eval()
    if size_loss:
        pred_raw, ori_pred_raw = model(x, edge_index)
    else:
        pred_raw = model(x, edge_index)

    if task == 'classification':
        pred = pred_raw.argmax(dim=-1)

    # print(data)
    accs = (f1_score(y[idx_test].cpu(), pred[idx_test].cpu(), average='micro'))
    # for mask in [train_mask, valid_mask, calib_test_mask]:
    #     if task == 'classification':
    #         accs.append(int((pred[mask] == y[mask]).sum()) / int(mask.sum()))
    #     # print(accs)

    if size_loss:
        if task == 'classification':
            out_softmax = F.softmax(pred_raw, dim=1)
            query_idx = np.where(valid_mask)[0]
            np.random.seed(0)
            np.random.shuffle(query_idx)

            train_train_idx = query_idx[:int(len(query_idx) / 2)]
            train_calib_idx = query_idx[int(len(query_idx) / 2):]

            n_temp = len(train_calib_idx)
            q_level = np.ceil((n_temp + 1) * (1 - alpha)) / n_temp

            tps_conformal_score = out_softmax[train_calib_idx][
                torch.arange(len(train_calib_idx)), y[train_calib_idx]]
            qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')
            c = torch.sigmoid((out_softmax[train_train_idx] - qhat) / tau)
            size_loss = torch.mean(torch.relu(torch.sum(c, axis=1) - target_size))

        return accs, pred_raw, size_loss.item()
    else:
        return accs, pred_raw


if __name__ == '__main__':
    tau2res = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    adj, features, one_hot_labels, ori_idx_train, idx_val, ori_idx_test = utils.load_data(args.dataset)

    # print("Number of True values in test_mask:", num_true_train, num_true_test, num_true_val, int(test_mask.shape[0] / 2))

    nx_g = nx.Graph(adj + sp.eye(adj.shape[0]))
    g = dgl.from_networkx(nx_g).to(device)
    # Get edge information (edge_index) from the DGL graph
    edge_index = g.edges()

    # If you need to work with a COO format
    # coo_edge_index = edge_index.t().contiguous()
    labels = torch.LongTensor([np.where(r == 1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]).to(device)
    features = torch.FloatTensor(utils.preprocess_features(features)).to(device)
    xent = nn.CrossEntropyLoss(reduction='none')
    # model = GNN(features.shape[1], hiddens, F.relu, feat_drop=0.5, edge_drop=0.5, alpha=0.1, k=12)

    for run in tqdm(range(args.epochs)): 
        result_this_run = {}
        if args.TRAININGSET == 'Baised':
            if not args.base_retrain:
                model_checkpoint_base = './model_saved/' + 'gnn_base' + '_' + args.dataset + '_' + 'baised' + '_' + '4242' + '_0410.pt'
            else:
                model_checkpoint_base = './model_saved/' + 'gnn_base' + '_' + args.dataset + '_' + 'baised' + '_' + str(
                    run + 1) + '_0410.pt'
        else:
            if args.base_retrain:
                model_checkpoint_base = './model_saved/' + 'gnn_base' + '_' + args.dataset + '_' + '4242' + '_0410.pt'
            else:
                model_checkpoint_base = './model_saved/' + 'gnn_base' + '_' + args.dataset + '_' + str(run + 1) + '_0410.pt'
        if args.model == 'APPNP':
            model = models.APPNP(g, features.shape[1], args.hiddens, labels.max().item() + 1, F.relu, args.k, feat_drop=0.5, edge_drop=0.5,
                      alpha=0.1)
        elif args.model == 'DAGNN':
            model = models.DAGNN(k=12, in_dim=features.shape[1], hid_dim=64, out_dim=labels.max().item() + 1)
        elif args.model == 'GCN':
            model = models.GCN_(g, features.shape[1], 32, labels.max().item() + 1, 1, F.tanh, 0.2)
        elif args.model == 'GAT':
            model = models.GAT_(features.shape[1], 8, labels.max().item() + 1, [8, 1])
        elif args.model == 'GraphSAGE':
            model = models.GraphSAGE(features.shape[1], 16, labels.max().item() + 1)
        optimiser = torch.optim.Adam(model.parameters(), args.lr, weight_decay=0.0005)
        # an example of biased training data
        # Generating PPR matrix for biased data generation
        # if arg.TRAININGSET == 'Biased'
        if args.BIASDATA == 'generate_own_bias_data':
            ppr_vector = torch.FloatTensor(utils.calc_ppr_exact(adj, args.alpha_bias))
            ppr_dist = utils.pairwise_distances(ppr_vector)
            train_dump = pickle.dump({'ppr_vector': ppr_vector, 'ppr_dist':ppr_dist},open('intermediate/{}_{}_dump.p'.format(args.dataset, args.alpha_bias), 'wb'))
            
            idx_train = utils.biased_Data(one_hot_labels, ori_idx_train, idx_val, ori_idx_test, args.dataset, ppr_dist,
                                          labels)
        else:
            if args.alpha_bias == 0.1:
                train_dump = pickle.load(open('intermediate/{}_{}_dump.p'.format(args.dataset, args.alpha_bias), 'rb'))
            # ppr_vector = train_dump['ppr_vector']
            # ppr_dist = train_dump['ppr_dist']
                idx_train = torch.LongTensor(pickle.load(open('data/localized_seeds_{}.p'.format(args.dataset), 'rb'))[0])
            elif args.alpha_bias == 0.2:
                train_dump = pickle.load(open('intermediate/{}_{}_dump.p'.format(args.dataset, args.alpha_bias), 'rb'))
            # ppr_vector = train_dump['ppr_vector']
                ppr_dist = train_dump['ppr_dist']
                idx_train = utils.biased_Data(one_hot_labels, ori_idx_train, idx_val, ori_idx_test, args.dataset, ppr_dist,
                                          labels)
            elif args.alpha_bias == 0.45:
                train_dump = pickle.load(open('intermediate/{}_{}_dump.p'.format(args.dataset, args.alpha_bias), 'rb'))
            # ppr_vector = train_dump['ppr_vector']
                ppr_dist = train_dump['ppr_dist']
                idx_train = utils.biased_Data(one_hot_labels, ori_idx_train, idx_val, ori_idx_test, args.dataset, ppr_dist,
                                          labels)
            elif args.alpha_bias == 0.65:
                train_dump = pickle.load(open('intermediate/{}_{}_dump.p'.format(args.dataset, args.alpha_bias), 'rb'))
            # ppr_vector = train_dump['ppr_vector']
                ppr_dist = train_dump['ppr_dist']
                idx_train = utils.biased_Data(one_hot_labels, ori_idx_train, idx_val, ori_idx_test, args.dataset, ppr_dist,
                                          labels)
        

        if False:
            # generating unbaised data (it create fluctuation in the accuracy)
            all_idx = set(range(g.number_of_nodes())) - set(idx_train)
            idx_test = torch.LongTensor(list(all_idx))
            perm = torch.randperm(idx_test.shape[0])
            iid_train = idx_test[perm[:idx_train.shape[0]]]
            pickle.dump({'iid_train': iid_train}, open('data_iid_train/{}_dump.p'.format(args.dataset), 'wb'))
        else:
            iid_train, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val, ori_idx_test,
                                                                max_train=20)
        label_balance_constraints = np.zeros((labels.max().item() + 1, len(idx_train)))
        for i, idx in enumerate(idx_train):
            label_balance_constraints[labels[idx], i] = 1
        all_idx = set(range(g.number_of_nodes()))
        idx_test = torch.LongTensor(list(all_idx))
        if args.TRAININGSET == 'Baised':
            train_mask = utils.sample_mask(idx_train, labels.shape[0])
            val_mask = utils.sample_mask(idx_val, labels.shape[0])
            test_mask = utils.sample_mask(idx_test, labels.shape[0])
        else:
            train_mask = utils.sample_mask(iid_train, labels.shape[0])
            val_mask = utils.sample_mask(idx_val, labels.shape[0])
            test_mask = utils.sample_mask(idx_test, labels.shape[0])
        n = min(1000, int(test_mask.shape[0] / 2))  # size of calibration set
        if args.size_loss:
            ######################################
            if args.conftr_calib_holdout:
                calib_test_idx = np.where(test_mask)[0]
                np.random.seed(run)
                np.random.shuffle(calib_test_idx)
                calib_eval_idx = calib_test_idx[:int(n * args.calib_fraction)]
                calib_test_real_idx = calib_test_idx[int(n * args.calib_fraction):]

                calib_eval_mask = np.array([False] * len(labels))
                calib_eval_mask[calib_eval_idx] = True
                calib_test_real_mask = np.array([False] * len(labels))
                calib_test_real_mask[calib_test_real_idx] = True
                # print('Using calibration distribution same as test distribution')
                calib_eval_idx = np.where(calib_eval_mask)[0]
                np.random.seed(run)
                np.random.shuffle(calib_eval_idx)
                train_calib_idx = calib_eval_idx[int(len(calib_eval_idx) / 2):]
                train_test_idx = calib_eval_idx[:int(len(calib_eval_idx) / 2)]

        if (os.path.exists(model_checkpoint_base)) and (not args.base_retrain):
            # used for applying CP as wrap-up for biased and iid data on pretrained model
            print('loading saved base model...')
            model = torch.load(model_checkpoint_base, map_location=device)
            # model, args.dataset = model.to(device), args.dataset.to(device)
            model.eval()
            embeds = model(features, edge_index).detach()
            best_model = model
            best_pred = embeds
            logits = best_pred[idx_test]
            preds_all = torch.argmax(best_pred, dim=1)
        else:
            # training base model
            print('training base model...')
            for epoch in range(args.epochs_base):
                model.train()
                optimiser.zero_grad()
                if args.model in ('APPNP', 'GCN'):
                    logits = model(features, edge_index)
                else:
                    logits = model(g, features)
                # logits = F.log_softmax(logits1, 1)
                if args.TRAININGSET == 'Baised':
                    loss = xent(logits[idx_train], labels[idx_train])
                else:
                    loss = xent(logits[iid_train], labels[iid_train])

                if args.size_loss:
                    out_softmax = F.softmax(logits, dim=1)
                    # ori_out_softmax = F.softmax(ori_out, dim=1)

                    n_temp = len(train_calib_idx)
                    q_level = np.ceil((n_temp + 1) * (1 - args.alpha_cov)) / n_temp

                    tps_conformal_score = out_softmax[train_calib_idx][
                        torch.arange(len(train_calib_idx)), labels[train_calib_idx]]
                    qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')

                    c = torch.sigmoid((out_softmax[train_test_idx] - qhat) / args.tau)
                    size_loss = torch.mean(torch.relu(torch.sum(c, axis=1)))

                if args.METHOD == 'SRGNN':
                    # print(out, ori_out)
                    if args.size_loss:
                        ########################################
                        loss = loss.mean() + 0.5 * model.shift_robust_output(idx_train, iid_train) \
                               + model.MMD_output(idx_train, iid_train) + size_loss
                    else:
                        loss = loss.mean() + 0.5 * model.shift_robust_output(idx_train, iid_train) \
                               + model.MMD_output(idx_train, iid_train)
                    # regularizer only: loss = loss.mean() + model.shift_robust_output(idx_train, iid_train)
                    # instance-reweighting only: loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()
                    # loss = (torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean() + model.shift_robust_output(idx_train,
                    #                                                                                          iid_train)
                elif args.METHOD == 'SRKL':
                    
                    if args.size_loss:
                        
                        loss = loss.mean() + 0.5 * model.shift_robust_output(idx_train, iid_train) \
                               + model.KLD_output(idx_train, iid_train) + size_loss
                    else:
                        loss = loss.mean() + 0.5 * model.shift_robust_output(idx_train, iid_train) \
                               + model.KLD_output(idx_train, iid_train)
                elif args.METHOD == 'SRJS':
                    
                    if args.size_loss:
                        loss = loss.mean() + 0.5 * model.shift_robust_output(idx_train, iid_train) \
                               + model.JSD_output(idx_train, iid_train) + size_loss
                    else:
                        loss = loss.mean() + 0.5 * model.shift_robust_output(idx_train, iid_train) \
                               + model.JSD_output(idx_train, iid_train)            
                elif args.METHOD == 'MMD':
                    if args.size_loss:
                        loss = loss.mean() + model.MMD_output(idx_train, iid_train) + size_loss
                    else:
                        loss = loss.mean() + model.MMD_output(idx_train, iid_train)
                elif args.METHOD == 'cmd':
                    if args.size_loss:
                        loss = loss.mean() + model.CMD_output(idx_train, iid_train) + size_loss
                    else:
                        loss = loss.mean() + model.CMD_output(idx_train, iid_train)        
                elif args.METHOD == 'kld':
                    if args.size_loss:
                        loss = loss.mean() + model.KLD_output(idx_train, iid_train) + size_loss
                    else:
                        loss = loss.mean() + model.KLD_output(idx_train, iid_train)
                elif args.METHOD == 'jsd':
                    if args.size_loss:
                        loss = loss.mean() + model.JSD_output(idx_train, iid_train) + size_loss
                    else:
                        loss = loss.mean() + model.JSD_output(idx_train, iid_train)
                elif args.METHOD == 'emd':
                    if args.size_loss:
                        loss = loss.mean() + model.EMD_output(idx_train, iid_train) + size_loss
                    else:
                        loss = loss.mean() + model.EMD_output(idx_train, iid_train)                            
                elif args.METHOD is None:
                    if args.size_loss:
                        loss = loss.mean() + size_loss
                    else:
                        loss = loss.mean()

                loss.backward()
                optimiser.step()

            model.eval()
            if args.model in ('APPNP', 'GCN'):
                embeds = model(features, edge_index).detach()
            else:
                embeds = model(g, features).detach()

            # embeds = model(features, edge_index).detach()
            best_pred = embeds
            logits = embeds[idx_test]
            preds_all = torch.argmax(embeds, dim=1)

            print("Accuracy:{}".format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')))
            # if args.TRAININGSET == 'Baised':
            #     # model_checkpoint_base = './model_saved/' + 'gnn_base' + '_' + args.dataset + '_' + 'baised' + '_' + str(
            #     #     run + 1) + '_0410.pt'
            #     if (f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')) >= 0.74:
            #         torch.save(model, model_checkpoint_base)
            # # elif not base_retrain:
            # #     torch.save(model, model_checkpoint_base_optimal)
            # else:
            #     # model_checkpoint_base = './model_saved/' + 'gnn_base' + '_' + args.dataset + '_' + str(run + 1) + '_0410.pt'
            #     if (f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')) >= 0.85:
            #         torch.save(model, model_checkpoint_base)
        # ---------- CP steps directly on prediction -----------
        result_this_run['Accuracy'] = f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')

        result_this_run['APS'] = run_conformal_classification(best_pred, labels, idx_test, n, args.alpha_cov,
                                                                     args.conformal_score,
                                                                     calib_eval=False)
        result_this_run['APSeps05'] = run_conformal_classification(best_pred, labels, idx_test, n, args.alpha_cov1,
                                                              args.conformal_score,
                                                              calib_eval=False)
        print(result_this_run['APS'])  # coverage & size based on the base model's prediction

        tau2res[run] = result_this_run
        # preds_all = torch.argmax(best_pred, dim=1)
        # # print(best_pred[calib_test])
        # print("Accuracy:{}".format(f1_score(y[calib_test], preds_all[calib_test], average='micro')))
        print(tau2res)
    if not os.path.exists('./pred'):
        os.mkdir('./pred')
    # if not args.not_save_res:
    #     print('Saving results to', './pred/' + name + '.pkl')
    #     with open('./pred/' + name + '.pkl', 'wb') as f:
    #         pickle.dump(tau2res, f)


    if not args.not_save_res:
        print('Saving results to', './pred/' + name + '.txt')
        with open('./pred/' + name + '.txt', 'w') as f:
            for result_this_run in tau2res.items():
                # f.write(f'Tau: {tau}\n')
                f.write(f'Result: {result_this_run}\n')
                f.write('\n')


parameter_set = {
            'confnn_hidden_dim': {'values': [16, 32, 64, 128, 256]},
            'confgnn_lr': {'values': [1e-1,1e-2,1e-3,1e-4]},
            'confgnn_num_layers': {'values': [1,2,3,4]},
            'confgnn_base_model': {'values': ['GAT', 'GCN', 'GraphSAGE', 'SGC']},
            'size_loss_weight': {'values': [1,1e-1,1e-2,1e-3]},
            'reg_loss_weight': {'values': [1,1e-1]}
         }