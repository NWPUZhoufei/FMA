import numpy as np
from io_utils import parse_args_test
import dataset_test
import torch
import scipy.stats as stats
import random
from torch.autograd import Variable
import torch.nn as nn
import vision_transformer as vits
import MAE_decoder
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def Tr_NN(support_set, query_set, pred, K):
        # support_set (way,shot,640)
        # query set (way*query,640)
        # tr就是每次根据query set的预测概率，从query set中找到置信度比较高的前K个样本，对support set进行更新。
        # 第一次迭代：每次迭代为每个类别抽取一个置信度最高的样本用于更新prototype。
        # pred （75，5）
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        score, index = pred.max(1)
        # step1：按照每个类别预测概率进行归类，统计每个类别的预测概率。
        all_class_score = []
        all_class_index = []
        for j in range(5):
            current_class_score = []
            current_class_index = []
            for i in range(75):
                if index[i]==j:
                    current_class_score.append(score[i])
                    current_class_index.append(i)
            all_class_score.append(current_class_score)
            all_class_index.append(current_class_index)
        
        # 选取置信度最高的K个样本
        prototype = []
        for i in range(5):
            current_class_index = all_class_index[i]
            if len(current_class_index) == 0:
                current_prototype = torch.mean(support_set[i],0) # (640)
            elif len(current_class_index) <= K:
                current_query_image = query_set[current_class_index] # (1,640)  
                current_prototype = torch.cat((support_set[i], current_query_image),0) # (shot+1,640)
                current_prototype = torch.mean(current_prototype, 0) # (640)
            else:
                current_class_score = all_class_score[i]
                # 首先对每个类别的预测中，选择两个置信度最高的。
                current_class_score_index = np.argsort(current_class_score)
                current_class_index = np.array(current_class_index)[current_class_score_index[-K:].tolist()] # 升序，选最后两个
                current_query_image = query_set[current_class_index] # (2,640) n>=2
                current_prototype = torch.cat((support_set[i], current_query_image),0) # (shot+2,640)
                current_prototype = torch.mean(current_prototype, 0) # (640)
            prototype.append(current_prototype)
        prototype = torch.stack(prototype, 0) #(5,640)
          
        return prototype
    
def Tr_LR(support_set, query_set, pred, K):
        # support_set (way,shot,640)
        # query set (way*query,640)
        # tr就是每次根据query set的预测概率，从query set中找到置信度比较高的前K个样本，对support set进行更新。
        # 第一次迭代：每次迭代为每个类别抽取一个置信度最高的样本用于更新prototype。
        # pred （75，5）
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        score, index = pred.max(1)
        # step1：按照每个类别预测概率进行归类，统计每个类别的预测概率。
        all_class_score = []
        all_class_index = []
        for j in range(5):
            current_class_score = []
            current_class_index = []
            for i in range(75):
                if index[i]==j:
                    current_class_score.append(score[i])
                    current_class_index.append(i)
            all_class_score.append(current_class_score)
            all_class_index.append(current_class_index)
        
        # 选取置信度最高的K个样本
        tr_support_set = []
        for i in range(5):
            current_class_index = all_class_index[i]
            if len(current_class_index) == 0:
                current_support_set = support_set[i] # (shot,640)
            elif len(current_class_index) <= K:
                current_query_image = query_set[current_class_index] # (1,640)  
                current_support_set = torch.cat((support_set[i], current_query_image),0) # (shot+k,640)
            else:
                current_class_score = all_class_score[i]
                # 首先对每个类别的预测中，选择两个置信度最高的。
                current_class_score_index = np.argsort(current_class_score)
                current_class_index = np.array(current_class_index)[current_class_score_index[-K:].tolist()] # 升序，选最后两个
                current_query_image = query_set[current_class_index] # (2,640) n>=2
                current_support_set = torch.cat((support_set[i], current_query_image),0) # (shot+K,640)
            
            tr_support_set.append(current_support_set)
        # 返回二维list，每个list里面是一个类别的tr之后的总的support样本个数
        tr_support_set_all = torch.cat((tr_support_set[0], tr_support_set[1], tr_support_set[2], tr_support_set[3], tr_support_set[4]),0) # (-,640)
        tr_support_set_gt = [0] * len(tr_support_set[0]) + [1] * len(tr_support_set[1]) + [2] * len(tr_support_set[2]) + [3] * len(tr_support_set[3]) + [4] * len(tr_support_set[4])
        tr_support_set_gt = np.array(tr_support_set_gt)
        
        return tr_support_set_all, tr_support_set_gt
    
def test(model, support_data, query_data, params):
   
    out_support, _, _, _ = model(support_data) # (way*shot,512)
    out_query, _, _, _ = model(query_data) # (way*query,512)
    
    out_support = out_support[:, 0]
    out_query = out_query[:, 0]
    
    # LR
    out_support_LR = out_support.cpu().numpy()
    out_query_LR = out_query.cpu().numpy()
    y = np.tile(range(params.n_way), params.n_support)
    y.sort()
    classifier = LogisticRegression(max_iter=1000).fit(X=out_support_LR, y=y)
    LR_pred = classifier.predict_proba(out_query_LR)
    LR_pred = torch.FloatTensor(LR_pred).cuda()
    y_query = np.repeat(range(params.n_way), params.n_query)
    topk_scores, topk_labels = LR_pred.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    correct_this, count_this = float(top1_correct), len(y_query)
    acc_LR = correct_this/ count_this *100
    
    # 下面为LR进行tr，按照所有数据的方式训练LR
    pred = LR_pred
    # 根据初始pred进行tr，对support set进行扩充
    out_support_LR = out_support.view(params.n_way, params.n_support, params.feature_size) #(way,shot,512) 
    for k in range(params.tr_N):
        tr_support_set, tr_support_set_gt = Tr_LR(out_support_LR, out_query, pred, params.tr_K)
        # tr_support_set是二维tensor，tr_support_set_gt是对应的标签是np
        # 根据更新后的support set继续计算pred
        tr_support_set = tr_support_set.cpu().numpy()
        # 使用LG分类器
        classifier = LogisticRegression(max_iter=1000).fit(X=tr_support_set, y=tr_support_set_gt)
        pred = classifier.predict_proba(out_query_LR)
        pred = torch.from_numpy(pred).cuda()
        
                
    topk_scores, topk_labels = pred.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    correct_this, count_this = float(top1_correct), len(y_query)
    acc_LR_tr = correct_this/ count_this *100
    
    

    return acc_LR, acc_LR_tr
    
    
def mini_batch_finetune_all(novel_loader, params):
    iter_num = len(novel_loader) 
    loss_fn = nn.CrossEntropyLoss().cuda()
    total_epoch = 100
    acc_all_LR = []
    acc_all_LR_tr = []
    
    for _, (x, y) in enumerate(novel_loader):
        x = x.cuda()
        x_var = Variable(x)
        batch_size = 4
        support_size = params.n_way*params.n_support 
        support_label = Variable(torch.from_numpy(np.repeat(range(params.n_way), params.n_support))).cuda() # (25,)
        query_data = x_var[:, params.n_support:,:,:,:].contiguous().view(params.n_way*params.n_query, *x.size()[2:]).cuda()
        support_data = x_var[:,:params.n_support,:,:,:].contiguous().view(params.n_way*params.n_support, *x.size()[2:]).cuda() # (25, 3, 224, 224)
        
        

        # 定义pretrain模型
        model_type = "dino"
        model_name = "vit_small"
        model_patchsize = 8
        model_p = vits.__dict__[model_name](mask_ratio=params.mask_ratio, patch_size=model_patchsize, num_classes=0)
        model_p.cuda()
        state_dict = torch.load(params.model_path)
        model_p.load_state_dict(state_dict, strict=True)
        print('model_type:', model_type)
        print(f"Model {model_name} {model_patchsize}x{model_patchsize} built.") 
        
        # 定义分类器
        classifier_p = Classifier(params.feature_size, params.n_way).cuda()
      
        
        # 定义pretrain模型
        model_type = "dino"
        model_name = "vit_small"
        model_patchsize = 8
        model_s = vits.__dict__[model_name](mask_ratio=params.mask_ratio, patch_size=model_patchsize, num_classes=0)
        model_s.cuda()
        print('model_type:', model_type)
        print(f"Model {model_name} {model_patchsize}x{model_patchsize} built.") 
        
        # 定义分类器
        classifier_s = Classifier(params.feature_size, params.n_way).cuda()
        
        # 定义mae decoder，共享参数的。
        mae_decoder = MAE_decoder.MAE_decoder(patch_size=model_patchsize, decoder_embed_dim=384).cuda()
        
        classifier_opt = torch.optim.SGD([{"params":classifier_p.parameters()}, {"params":classifier_s.parameters()}], lr = params.lr)
        classifier_p.train()    
        classifier_s.train()
            
        
        # 定义优化器
        delta_opt = torch.optim.SGD([{"params":mae_decoder.parameters()}, {"params":model_p.parameters()}, {"params":model_s.parameters()}], lr = params.lr)
        model_p.train()    
        model_s.train()
        mae_decoder.train() 
        
        loss_fn = nn.CrossEntropyLoss().cuda()
        

        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)
            for j in range(0, support_size, batch_size):
                delta_opt.zero_grad()
                classifier_opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                z_batch = support_data[selected_id]
                y_batch = support_label[selected_id] 
                
                
                out_p_raw, out_p_masked, out_p_mask, out_p_ids_restore  = model_p(z_batch) # (_, 1+196,768)、(_, 1+196-M,768)
                pred_p_raw = classifier_p(out_p_raw[:,0]) 
                out_s_raw, out_s_masked, out_s_mask, out_s_ids_restore = model_s(z_batch) # (_, 1+196,768)、(_, 1+196-M,768)
                pred_s_raw = classifier_s(out_s_raw[:,0])
                
                # 进行重建，对称的
                mae_p = mae_decoder(out_p_masked, out_p_ids_restore) # # 返回所有的patch嵌入(-,196,768)，不包括class token
               
                # mask : torch.Size([-, 196]) 用于计算mae重建loss
                # ids_restore : (-, 196)
                
                
               
                # 下面计算ce loss
                loss_ce_p = loss_fn(pred_p_raw, y_batch)
                loss_ce_s = loss_fn(pred_s_raw, y_batch)
                
                # 计算mae loss
                loss_p = (mae_p - out_s_raw[:,1:,:]) ** 2
                loss_p = loss_p.mean(dim=-1)  # [N, L], mean loss per patch
                loss_p = (loss_p * out_p_mask).sum() / out_p_mask.sum()  # mean loss on removed patches
                
                # 下面计算自相似性正则损失
                out_p_class_token = out_p_raw[:,0]
                out_p_local_token = out_p_raw[:,1:,:]
                N, L ,D = out_p_local_token.size()
                out_p_class_token = out_p_class_token.unsqueeze(1).expand(N, L, D)
                p_ss = F.cosine_similarity(out_p_class_token, out_p_local_token, dim=2) 
                
                out_s_class_token = out_s_raw[:,0]
                out_s_local_token = out_s_raw[:,1:,:]
                N, L ,D = out_s_local_token.size()
                out_s_class_token = out_s_class_token.unsqueeze(1).expand(N, L, D)
                s_ss = F.cosine_similarity(out_s_class_token, out_s_local_token, dim=2) 
                
                loss_ss = (p_ss - s_ss) ** 2
                loss_ss = loss_ss.mean() 
                
        
                # 下面计算总loss
                loss = loss_ce_p + loss_ce_s + loss_p * params.lambda_mae_loss + loss_ss * params.lambda_ss_loss
                
                loss.backward()
                delta_opt.step()
                classifier_opt.step()
       
        # 下面进行推理测试：
        with torch.no_grad():
            acc_LR, acc_LR_tr = test(model_p, support_data, query_data, params)
        
       
        acc_all_LR.append(acc_LR)
        acc_all_LR_tr.append(acc_LR_tr)

   
    acc = np.asarray(acc_all_LR)
    acc_mean = np.mean(acc)
    acc_std  = np.std(acc)
    print('%d LR : %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    
    acc = np.asarray(acc_all_LR_tr)
    acc_mean = np.mean(acc)
    acc_std  = np.std(acc)
    print('%d LR_tr : %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    


    
    
    
if __name__=='__main__':
    
    params = parse_args_test()
    setup_seed(params.seed)
    
    datamgr = dataset_test.Eposide_DataManager(data_path=params.current_data_path, num_class=params.current_class, image_size=params.image_size, n_way=params.n_way, n_support=params.n_support, n_query=params.n_query, n_eposide=params.test_n_eposide)
    novel_loader = datamgr.get_data_loader(aug=False) 
    
    mini_batch_finetune_all(novel_loader, params)
    
        
        
 
 
    
    