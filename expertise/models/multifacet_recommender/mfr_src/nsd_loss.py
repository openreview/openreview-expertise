import torch
import random
from .model import MatrixReconstruction as MR

def predict_basis(model_set, n_basis, output_emb, predict_coeff_sum = False):
    #print( output_emb.size() )
    #output_emb should have dimension ( n_batch, n_emb_size)

    if predict_coeff_sum:
        basis_pred, coeff_pred =  model_set(output_emb, predict_coeff_sum = True)
        #basis_pred should have dimension ( n_basis, n_batch, n_emb_size)
        #coeff_pred should have dimension ( n_basis, n_batch, 2)

        #basis_pred = basis_pred.permute(1,0,2)
        #coeff_pred = coeff_pred.permute(1,0,2)
        #basis_pred should have dimension ( n_batch, n_basis, n_emb_size)
        return basis_pred, coeff_pred
    else:
        basis_pred =  model_set(output_emb, predict_coeff_sum = False)
        #basis_pred = basis_pred.permute(1,0,2)
        return basis_pred

def estimate_coeff_mat_batch_max_iter(target_embeddings, basis_pred, device):
    batch_size = target_embeddings.size(0)
    #A = basis_pred.permute(0,2,1)
    C = target_embeddings.permute(0,2,1)
    #basis_pred_norm = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
    
    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
    #basis_pred_norm_sq = basis_pred_norm * basis_pred_norm
    XX = basis_pred_norm * basis_pred_norm
    n_not_sparse = 2
    coeff_mat_trans = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
    for i in range(n_not_sparse):
        XY = torch.bmm(basis_pred, C)
        coeff = XY / XX
        #coeff should have dimension ( n_batch, n_basis, n_set)
        max_v, max_i = torch.max(coeff, dim = 1, keepdim=True)
        max_v[max_v<0] = 0
    
        coeff_mat_trans_temp = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
        coeff_mat_trans_temp.scatter_(dim=1, index = max_i, src = max_v)
        coeff_mat_trans.scatter_add_(dim=1, index = max_i, src = max_v)
        #pred_emb = torch.bmm(coeff_mat_trans_temp.permute(0,2,1),basis_pred)
        #C = C - pred_emb
        pred_emb = torch.bmm(coeff_mat_trans.permute(0,2,1),basis_pred)
        C = (target_embeddings - pred_emb).permute(0,2,1)
        
    #pred_emb = max_v * torch.gather(basis_pred,  max_i
    
    return coeff_mat_trans.permute(0,2,1)
    #torch.gather(coeff_mat_trans , dim=1, index = max_i)

def estimate_coeff_mat_batch_max_cos(target_embeddings, basis_pred):
    #batch_size = target_embeddings.size(0)
    C = target_embeddings.permute(0,2,1)

    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
    XX = basis_pred_norm * basis_pred_norm
    XY = torch.bmm(basis_pred, C)
    coeff = XY / XX
    #coeff should have dimension ( n_batch, n_basis, n_set)
    max_v, max_i = torch.max(coeff, dim = 1)
    return max_v, max_i

def estimate_coeff_mat_batch_softmax(target_embeddings, basis_pred, device, loss_type='dist', target_norm=False):
    batch_size = target_embeddings.size(0)
    n_basis = basis_pred.size(1)
    if n_basis == 1 and loss_type[:4] != 'dist':
        coeff_mat = torch.ones(batch_size, target_embeddings.size(1), n_basis, requires_grad= False, device=device )
        return coeff_mat

    if target_norm:
        #print(target_embeddings.norm(dim=2))
        target_embeddings_norm = target_embeddings / (0.000000000001 + target_embeddings.norm(dim = 2, keepdim=True) ) # If this step is really slow, consider to do normalization before doing unfold
    else:
        target_embeddings_norm = target_embeddings
    C = target_embeddings_norm.permute(0,2,1)

    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
    #XX = basis_pred_norm * basis_pred_norm
    XY = torch.bmm(basis_pred, C)
    cos_sim = XY / basis_pred_norm
    #T = 0.2
    T = 1
    sim_softmax = torch.nn.functional.softmax(cos_sim / T, dim = 1)
    return sim_softmax.permute(0,2,1)


def estimate_coeff_mat_batch_max(target_embeddings, basis_pred, device, loss_type='dist', target_norm=False,
                                 always_norm_one=False):
    batch_size = target_embeddings.size(0)
    n_basis = basis_pred.size(1)
    if n_basis == 1 and loss_type[:4] != 'dist':
        coeff_mat = torch.ones(batch_size, target_embeddings.size(1), n_basis, requires_grad= False, device=device )
        return coeff_mat

    if target_norm:
        #print(target_embeddings.norm(dim=2))
        target_embeddings_norm = target_embeddings / (0.000000000001 + target_embeddings.norm(dim = 2, keepdim=True) ) # If this step is really slow, consider to do normalization before doing unfold
    else:
        target_embeddings_norm = target_embeddings
    C = target_embeddings_norm.permute(0,2,1)

    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
    XX = basis_pred_norm * basis_pred_norm
    XY = torch.bmm(basis_pred, C)
    if loss_type[:4] == 'dist':
        cos_sim = XY / basis_pred_norm
        max_v_cos, max_i_cos = torch.max(cos_sim, dim = 1, keepdim=True)
        coeff = XY / XX
        ##coeff should have dimension ( n_batch, n_basis, n_set)
        ##max_v, max_i = torch.max(coeff, dim = 1, keepdim=True)
        
        max_v = torch.gather(coeff, dim=1, index=max_i_cos)
        if always_norm_one:
            # max_v[max_v!=0] = 1
            norm_max = torch.gather(basis_pred_norm.expand(batch_size, n_basis, cos_sim.size(2)), dim=1,
                                    index=max_i_cos)
            max_v = 1 / norm_max
            # max_v[max_v>1] = 1
        else:
            max_v[max_v < 0] = 0
            max_v[max_v > 1] = 1

        #max_v = torch.ones(max_v_cos.size(), requires_grad= False, device=device)
        
        #norm_max = torch.gather(basis_pred_norm.expand(batch_size,n_basis, cos_sim.size(2)), dim = 1, index = max_i_cos)
        #max_v = 1 / norm_max
        #max_v[max_v>1] = 1
    else:
        max_v_cos, max_i_cos = torch.max(XY, dim = 1, keepdim=True)
        max_v = torch.ones(max_v_cos.size(), requires_grad= False, device=device)
        max_v[max_v_cos<0] = 0

    coeff_mat_trans = torch.zeros(batch_size, n_basis, target_embeddings.size(1), requires_grad= False, device=device )
    coeff_mat_trans.scatter_(dim=1, index = max_i_cos, src = max_v)
    return coeff_mat_trans.permute(0,2,1)

#def estimate_coeff_mat_batch_max(target_embeddings, basis_pred, device):
#    batch_size = target_embeddings.size(0)
#    C = target_embeddings.permute(0,2,1)
#    
#    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
#    XX = basis_pred_norm * basis_pred_norm
#    XY = torch.bmm(basis_pred, C)
#    coeff = XY / XX
#    #coeff should have dimension ( n_batch, n_basis, n_set)
#    max_v, max_i = torch.max(coeff, dim = 1, keepdim=True)
#    max_v[max_v<0] = 0
#    
#    coeff_mat_trans = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
#    coeff_mat_trans.scatter_(dim=1, index = max_i, src = max_v)
#    return coeff_mat_trans.permute(0,2,1)

def estimate_prod_coeff_mat_batch_opt(target_embeddings, basis_pred, L1_losss_B, device, coeff_opt, lr, max_iter):
    batch_size = target_embeddings.size(0)
    mr = MR(batch_size, target_embeddings.size(1), basis_pred.size(1), device=device)
    if coeff_opt == 'sgd':
        opt = torch.optim.SGD(mr.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    elif coeff_opt == 'asgd':
        opt = torch.optim.ASGD(mr.parameters(), lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    elif coeff_opt == 'adagrad':
        opt = torch.optim.Adagrad(mr.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    elif coeff_opt == 'rmsprop':
        opt = torch.optim.RMSprop(mr.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,centered=False)
    elif coeff_opt == 'adam':
        opt = torch.optim.Adam(mr.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        raise RuntimeError('%s not implemented for coefficient estimation. Please check args.' % coeff_opt)
    
    for i in range(max_iter):
        opt.zero_grad()
        pred = mr(basis_pred)
        loss = - torch.sum(pred * target_embeddings)
        #print(loss)
        loss += L1_losss_B * mr.coeff.abs().sum() 
        loss.backward()
        opt.step()
        mr.compute_coeff_pos_norm()
        #print(mr.coeff)
    #sys.exit(0)
    return mr.coeff.detach()

def estimate_coeff_mat_batch_opt(target_embeddings, basis_pred, L1_losss_B, device, coeff_opt, lr, max_iter):
    batch_size = target_embeddings.size(0)
    mr = MR(batch_size, target_embeddings.size(1), basis_pred.size(1), device=device)
    loss_func = torch.nn.MSELoss(reduction='sum')
    
    # opt = torch.optim.LBFGS(mr.parameters(), lr=lr, max_iter=max_iter, max_eval=None, tolerance_grad=1e-05,
    #                         tolerance_change=1e-09, history_size=100, line_search_fn=None)
    #
    # def closure():
    #     opt.zero_grad()
    #     mr.compute_coeff_pos()
    #     pred = mr(basis_pred)
    #     loss = loss_func(pred, target_embeddings) / 2
    #     # loss += L1_losss_B * mr.coeff.abs().sum()
    #     loss += L1_losss_B * (mr.coeff.abs().sum() + mr.coeff.diagonal(dim1=1, dim2=2).abs().sum())
    #     # print('loss:', loss.item())
    #     loss.backward()
    #
    #     return loss
    #
    # opt.step(closure)
    
    if coeff_opt == 'sgd':
        opt = torch.optim.SGD(mr.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    elif coeff_opt == 'asgd':
        opt = torch.optim.ASGD(mr.parameters(), lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    elif coeff_opt == 'adagrad':
        opt = torch.optim.Adagrad(mr.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    elif coeff_opt == 'rmsprop':
        opt = torch.optim.RMSprop(mr.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                                  centered=False)
    elif coeff_opt == 'adam':
        opt = torch.optim.Adam(mr.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        raise RuntimeError('%s not implemented for coefficient estimation. Please check args.' % coeff_opt)
    
    for i in range(max_iter):
        opt.zero_grad()
        pred = mr(basis_pred)
        loss = loss_func(pred, target_embeddings) / 2
        # loss += L1_losss_B * mr.coeff.abs().sum()
        #loss += L1_losss_B * (mr.coeff.abs().sum() + mr.coeff.diagonal(dim1=1, dim2=2).abs().sum())
        loss += L1_losss_B * mr.coeff.abs().sum() 
        # print('loss:', loss.item())
        loss.backward()
        opt.step()
        mr.compute_coeff_pos()
    
    return mr.coeff.detach()


def estimate_coeff_mat_batch(target_embeddings, basis_pred, L1_losss_B, device, max_iter = 100):
    def compute_matrix_magnitude(M_diff):
        return torch.mean( torch.abs(M_diff) )

    def update_B_from_AC(AT,BT,CT,A,lr):
        BT_grad = torch.bmm( torch.bmm(BT, AT) - CT, A )
        #BT = BT - lr * (BT_grad)
        #return BT, BT_grad
        BT = BT - lr * (BT_grad + L1_losss_B)

        BT_nonneg = BT.clamp(0,1)
        return BT_nonneg, BT_grad

    batch_size = target_embeddings.size(0)
    #converge_threshold = 0.01

    A = basis_pred.permute(0,2,1)
    #coeff_mat_prev = torch.abs(torch.randn(batch_size, target_embeddings.size(1), basis_pred.size(1), requires_grad= False, device=device ))
    coeff_mat_prev = torch.randn(batch_size, target_embeddings.size(1), basis_pred.size(1), requires_grad= False, device=device )
    #coeff_mat = update_B_from_AC(basis_pred, coeff_mat_prev, target_embeddings, A)
    #coeff_mat_prev = torch.zeros_like(coeff_mat)
    #max_iter = 50
    #max_iter = 150
    #max_iter = 100
    #max_iter = 200
    #diff_prev = 10
    lr = 0.05
    #lr = 0.02
    #lr = 0.2
    #lr = 0.02
    #lr = 0.1
    for i in range(max_iter):
        coeff_mat, coeff_mat_grad = update_B_from_AC(basis_pred, coeff_mat_prev, target_embeddings, A, lr)
        ##diff = compute_matrix_magnitude(coeff_mat - coeff_mat_prev)

        #diff = compute_matrix_magnitude(coeff_mat_grad)

        #print(diff)
        #if diff > diff_prev:
        #    lr *= 0.9
        #if diff < converge_threshold:
        #    break
        #diff_prev = diff
        coeff_mat_prev = coeff_mat
    #print(diff,lr,i)
    #coeff_mat should have dimension (n_batch,n_set,n_basis)
    return coeff_mat

def target_emb_preparation(target_index, embeddings, n_batch, n_set, target_index_shuffle, rand_neg_method, target_norm, target_linear_layer):
    target_embeddings = embeddings[target_index,:]
    if target_linear_layer is not None:
        target_embeddings = torch.bmm(target_embeddings, target_linear_layer.expand(target_embeddings.size(0),target_linear_layer.size(0), target_linear_layer.size(1)))
        #target_embeddings = target_embeddings.mean(dim=1).unsqueeze(dim=1)
    #print( target_embeddings.size() )
    #target_embeddings should have dimension (n_batch, n_set, n_emb_size)
    #should be the same as embeddings.select(0,target_set) and select should not copy the data
    if target_norm:
        target_embeddings = target_embeddings / (0.000000000001 + target_embeddings.norm(dim = 2, keepdim=True) ) # If this step is really slow, consider to do normalization before doing unfold
    
    #target_embeddings_4d = target_embeddings.view(-1,n_batch, n_set, target_embeddings.size(2))
    if rand_neg_method == 'rotate':
        target_embeddings_rotate = torch.cat( (target_embeddings[target_index_shuffle:,:,:], target_embeddings[:target_index_shuffle,:,:]), dim = 0)
    else:
        #target_index_shuffle = target_index.view(-1)[shuffle_idx].view(target_index.size())
        target_embeddings_rotate = embeddings[target_index_shuffle,:] 
        #TODO: Should not need to do the multiplication and normalization twice
        if target_linear_layer is not None:
            target_embeddings_rotate = torch.bmm(target_embeddings_rotate, target_linear_layer.expand(target_embeddings.size(0),target_linear_layer.size(0), target_linear_layer.size(1)))
        if target_norm:
            target_embeddings_rotate = target_embeddings_rotate / (0.000000000001 + target_embeddings_rotate.norm(dim = 2, keepdim=True) ) # If this step is really slow, consider to do normalization before doing unfold

    #target_emb_neg = target_embeddings_rotate.view(-1,n_set, target_embeddings.size(2))

    #return target_embeddings, target_emb_neg
    return target_embeddings, target_embeddings_rotate

#def compute_loss_set(output_emb, model_set, w_embeddings, target_set, n_basis, L1_losss_B, device, w_freq, coeff_opt, compute_target_grad):
#def compute_loss_set(output_emb, basis_pred, coeff_pred, entpair_embs, target_set, L1_losss_B, device, w_freq, w_freq_sampling, repeat_num, target_len, coeff_opt, loss_type, compute_target_grad, coeff_opt_algo, rand_neg_method, target_norm, compute_div_reg = True):
def compute_loss_set(basis_pred, entpair_embs, target_set, L1_losss_B, device, w_freq, w_freq_sampling, repeat_num, target_len, coeff_opt, loss_type, compute_target_grad, coeff_opt_algo, rand_neg_method, target_norm, compute_div_reg = True, target_linear_layer = None, pre_avg = False, always_norm_one = False):
    def compute_target_freq_inv_norm(w_freq, target_set):
        target_freq = w_freq[target_set]
        #target_freq = torch.masked_select( target_freq, target_freq.gt(0))
        target_freq_inv = 1 / target_freq
        target_freq_inv[target_freq_inv<0] = 0 #handle null case
        inv_mean = torch.sum(target_freq_inv) / torch.sum(target_freq_inv>0).float()
        if inv_mean > 0:
            target_freq_inv_norm =  target_freq_inv / inv_mean
        else:
            target_freq_inv_norm =  target_freq_inv
        return target_freq_inv_norm
    #basis_pred, coeff_pred = predict_basis(model_set, n_basis, output_emb, predict_coeff_sum = True)
    #basis_pred should have dimension ( n_batch, n_basis, n_emb_size)
    #print( basis_pred.size() )
    #print( target_set.size() )
    #target_set should have dimension (n_batch, n_set)

    n_set = target_set.size(1)
    n_batch = target_set.size(0)
    #rand_neg_method = 'rotate'
    #rand_neg_method = 'shuffle'
    #rand_neg_method = 'uniform'
    #rand_neg_method = 'paper_uniform'
    if rand_neg_method == 'paper_uniform':
        target_index_shuffle = torch.multinomial( w_freq_sampling, target_set.nelement(), replacement=True ).view(target_set.size())
        #shuffle might just have too many randomness (some paper might just get no negative samples). This might make the paper prior unstable.
    elif rand_neg_method == 'uniform':
        mask = torch.ge(target_set,1)
        target_index_shuffle = torch.randint_like(target_set, low = 1, high = w_freq.size(0), device=device)
        target_index_shuffle *= mask
    elif rand_neg_method == 'shuffle':
        shuffle_idx = torch.randperm(target_set.nelement())
        target_index_shuffle = target_set.view(-1)[shuffle_idx].view(target_set.size())
    elif rand_neg_method == 'rotate':
        #rotate_shift = random.randint(1,n_batch-1)
        #shuffle_idx = random.randint(1,n_batch-1)
        target_index_shuffle = random.randint(1,n_batch-1)

    if compute_target_grad:
        target_embeddings, target_emb_neg = target_emb_preparation(target_set, entpair_embs, n_batch, n_set, target_index_shuffle, rand_neg_method, target_norm, target_linear_layer)
    else:
        with torch.no_grad():
            target_embeddings, target_emb_neg = target_emb_preparation(target_set, entpair_embs, n_batch, n_set, target_index_shuffle, rand_neg_method, target_norm, target_linear_layer)
    #print( target_embeddings.size() )

    with torch.no_grad():
        target_freq_inv_norm = compute_target_freq_inv_norm(w_freq, target_set)
        if rand_neg_method == 'paper_uniform':
            target_freq_inv_norm_neg = compute_target_freq_inv_norm(w_freq, target_index_shuffle)
            neg_total_weights = target_len.sum() / float(target_freq_inv_norm_neg.nelement())
            repeat_num_inv = 1 / repeat_num.float()
            repeat_num_inv_norm = repeat_num_inv / repeat_num_inv.mean()
            target_freq_inv_norm_neg *= neg_total_weights * repeat_num_inv_norm.unsqueeze(dim=1)
            #remember to normalize
        elif rand_neg_method == 'uniform':
            target_freq_inv_norm_neg = compute_target_freq_inv_norm(w_freq, target_index_shuffle)
        elif rand_neg_method == 'shuffle':
            target_freq_inv_norm_neg = target_freq_inv_norm.view(-1)[shuffle_idx].view(target_freq_inv_norm.size())
        elif rand_neg_method == 'rotate':
            target_freq_inv_norm_neg = torch.cat( (target_freq_inv_norm[target_index_shuffle:,:], target_freq_inv_norm[:target_index_shuffle,:]), dim = 0)
        if pre_avg:
            #target_embeddings = (target_embeddings*target_freq_inv_norm.unsqueeze(dim=-1)).mean(dim=1).unsqueeze(dim=1)
            #target_emb_neg = (target_emb_neg*target_freq_inv_norm_neg.unsqueeze(dim=-1)).mean(dim=1).unsqueeze(dim=1)
            with torch.enable_grad():
                target_embeddings = ( (target_embeddings*target_freq_inv_norm.unsqueeze(dim=-1)).sum(dim=1) / (0.000000000001 + target_freq_inv_norm.sum(dim=1).unsqueeze(dim=-1) ) ).unsqueeze(dim=1)
                target_emb_neg = ( (target_emb_neg*target_freq_inv_norm_neg.unsqueeze(dim=-1)).sum(dim=1) / (0.000000000001 + target_freq_inv_norm_neg.sum(dim=1).unsqueeze(dim=-1) ) ).unsqueeze(dim=1)
            target_freq_inv_norm = 1
            target_freq_inv_norm_neg = 1
        #coeff_mat = estimate_coeff_mat_batch(target_embeddings.cpu(), basis_pred.detach(), L1_losss_B)
        if coeff_opt == 'prod':
            lr_coeff = 0.05
            iter_coeff = 60
            #iter_coeff = 100
            with torch.enable_grad():
                coeff_mat = estimate_prod_coeff_mat_batch_opt(target_embeddings.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
                coeff_mat_neg = estimate_prod_coeff_mat_batch_opt(target_emb_neg.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
            
        elif coeff_opt == 'lc':
            if coeff_opt_algo == 'sgd_bmm':
                coeff_mat = estimate_coeff_mat_batch(target_embeddings.detach(), basis_pred.detach(), L1_losss_B, device)
                coeff_mat_neg = estimate_coeff_mat_batch(target_emb_neg.detach(), basis_pred.detach(), L1_losss_B, device)
            else:
                lr_coeff = 0.05
                iter_coeff = 60
                with torch.enable_grad():
                    coeff_mat = estimate_coeff_mat_batch_opt(target_embeddings.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
                    coeff_mat_neg = estimate_coeff_mat_batch_opt(target_emb_neg.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
        elif coeff_opt == 'max':
            coeff_mat = estimate_coeff_mat_batch_max(target_embeddings.detach(), basis_pred.detach(), device, loss_type, not target_norm, always_norm_one)
            #coeff_mat = estimate_coeff_mat_batch_max_iter(target_embeddings, basis_pred.detach(), device)
            coeff_mat_neg = estimate_coeff_mat_batch_max(target_emb_neg.detach(), basis_pred.detach(), device, loss_type, not target_norm, always_norm_one)
        elif coeff_opt == 'softmax':
            coeff_mat = estimate_coeff_mat_batch_softmax(target_embeddings.detach(), basis_pred.detach(), device, loss_type, not target_norm)
            coeff_mat_neg = estimate_coeff_mat_batch_softmax(target_emb_neg.detach(), basis_pred.detach(), device, loss_type, not target_norm)

    #if coeff_opt == 'lc' and  coeff_opt_algo != 'sgd_bmm':
    #    lr_coeff = 0.05
    #    iter_coeff = 60
    #    coeff_mat = estimate_coeff_mat_batch_opt(target_embeddings.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
    #    coeff_mat_neg = estimate_coeff_mat_batch_opt(target_emb_neg.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
    
    pred_embeddings = torch.bmm(coeff_mat, basis_pred)
    pred_embeddings_neg = torch.bmm(coeff_mat_neg, basis_pred)
    #pred_embeddings should have dimension (n_batch, n_set, n_emb_size)
    #loss_set = torch.mean( target_freq_inv_norm * torch.norm( pred_embeddings.cuda() - target_embeddings, dim = 2 ) )
    if loss_type == 'sg':
        loss_set = - torch.mean( target_freq_inv_norm * torch.log( torch.sigmoid( torch.sum(pred_embeddings * target_embeddings, dim=2) ) ) )
        loss_set_neg = - torch.mean( target_freq_inv_norm_neg * torch.log(torch.sigmoid( - torch.sum(pred_embeddings_neg * target_emb_neg, dim=2) ) ) )
    elif loss_type == 'sim':
        loss_set = torch.mean( target_freq_inv_norm * torch.pow(1 - torch.sum(pred_embeddings * target_embeddings, dim=2), 2) )
        loss_set_neg = torch.mean( target_freq_inv_norm_neg * torch.pow( 0 -  torch.sum(pred_embeddings_neg * target_emb_neg, dim=2), 2) )
    elif loss_type == 'dist':
        loss_set = torch.mean( target_freq_inv_norm * torch.pow( torch.norm( pred_embeddings - target_embeddings, dim = 2 ), 2) )
        loss_set_neg = - torch.mean( target_freq_inv_norm_neg * torch.pow( torch.norm( pred_embeddings_neg - target_emb_neg, dim = 2 ), 2) )
    elif loss_type == 'dist_expand':
        loss_set = torch.mean( target_freq_inv_norm * ( torch.sum(pred_embeddings * pred_embeddings, dim=2) - 2 * torch.sum(pred_embeddings * target_embeddings, dim=2) ) )
        loss_set_neg = - torch.mean( target_freq_inv_norm_neg * ( torch.sum(pred_embeddings_neg * pred_embeddings_neg, dim=2) - 2* torch.sum(pred_embeddings_neg * target_emb_neg, dim=2) ) )
    
    #if coeff_pred != None:
    #    basis_pred_mag = basis_pred.norm(dim = 2)
    #    with torch.no_grad():
    #        coeff_sum_basis = coeff_mat.sum(dim = 1)  / basis_pred_mag
    #        coeff_sum_basis_neg = coeff_mat_neg.sum(dim = 1)  / basis_pred_mag
    #        coeff_mean = (coeff_sum_basis.mean() + coeff_sum_basis_neg.mean()) / 2
    #        #coeff_sum_basis should have dimension (n_batch,n_basis)
    #    
    #    loss_coeff_pred = torch.mean( torch.pow( coeff_sum_basis/coeff_mean - coeff_pred[:,:,0].view_as(coeff_sum_basis), 2 ) )
    #    loss_coeff_pred += torch.mean( torch.pow( coeff_sum_basis_neg/coeff_mean - coeff_pred[:,:,1].view_as(coeff_sum_basis_neg), 2 ) )
        
    #if random.randint(0,n_batch) == 1:
    #    print("coeff_sum_basis/coeff_mean", coeff_sum_basis/coeff_mean )
    #    print("coeff_sum_basis", coeff_sum_basis[0,:] )
    #    #print("target_freq_inv_norm", target_freq_inv_norm )
    #    print("pred_embeddings", pred_embeddings[0,:,:] )
    #    print("target_embeddings", target_embeddings[0,:,:] )
    #    print("target_set", target_set[0,:])

    if torch.isnan(loss_set):
        #print("output_embeddings", output_emb.norm(dim = 1))
        print("basis_pred", basis_pred.norm(dim = 2))
        #print("coeff_sum_basis", coeff_sum_basis)
        print("pred_embeddings", pred_embeddings.norm(dim = 2) )
        print("target_embeddings", target_embeddings.norm(dim = 2) )

    if compute_div_reg:
        basis_pred_norm = basis_pred / basis_pred.norm(dim = 2, keepdim=True)
        with torch.no_grad():
            pred_mean = basis_pred_norm.mean(dim = 0, keepdim = True)
            loss_set_reg = - torch.mean( (basis_pred_norm - pred_mean).norm(dim = 2) )
    
        pred_mean = basis_pred_norm.mean(dim = 1, keepdim = True)
        loss_set_div = - torch.mean( (basis_pred_norm - pred_mean).norm(dim = 2) )

        if target_norm:
            target_embeddings_norm = target_embeddings
            target_emb_neg_norm = target_emb_neg
        else:
            target_embeddings_norm = target_embeddings / (0.000000000001 + target_embeddings.norm(dim = 2, keepdim=True) )
            target_emb_neg_norm = target_emb_neg / (0.000000000001 + target_emb_neg.norm(dim = 2, keepdim=True) )
        mask = (target_set > 0).unsqueeze(dim=-1)
        pred_mean = target_embeddings_norm.sum(dim = 1, keepdim = True) / (0.000000000001 + torch.sum(mask, dim=1, keepdim=True).to(device=target_embeddings_norm.device) ).detach()
        loss_set_div_target = - torch.sum( ((target_embeddings_norm - pred_mean) * mask).norm(dim = 2) ) / (0.000000000001 + torch.sum(mask))
        assert rand_neg_method == 'shuffle'
        mask = (target_index_shuffle > 0).unsqueeze(dim=-1)
        pred_mean = target_emb_neg_norm.sum(dim = 1, keepdim = True) / (0.000000000001 + torch.sum(mask, dim=1, keepdim=True).to(device=target_emb_neg_norm.device) ).detach()
        loss_set_div_target += - torch.sum( ((target_emb_neg_norm - pred_mean) * mask).norm(dim = 2) ) / (0.000000000001 + torch.sum(mask))

    if not compute_div_reg:
        return loss_set, loss_set_neg
    else:
        return loss_set, loss_set_neg, loss_set_div, loss_set_reg, loss_set_div_target
    #if coeff_pred == None:
    #    if not compute_div_reg:
    #        return loss_set, loss_set_neg
    #    else:
    #        return loss_set, loss_set_neg, loss_set_div, loss_set_reg, loss_set_div_target
    #else:
    #    if not compute_div_reg:
    #        return loss_set, loss_set_neg, loss_coeff_pred
    #    else:
    #        return loss_set, loss_set_neg, loss_set_div, loss_set_reg, loss_set_div_target, loss_coeff_pred
