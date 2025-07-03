import argparse
import torch
from dataloader import load_data
from classifier import LinearClassifier
import torch.nn as nn
import random
import numpy as np
import os
import utils
import vision_transformer as vits
import MAE_decoder
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(train_loader, model_p, classifier_p, model_s, classifier_s, mae_decoder, loss_fn, optimizer, args):
    top1 = utils.AverageMeter()
    total_loss = 0
    model_p.train()
    classifier_p.train()
    model_s.train()
    classifier_s.train()
    mae_decoder.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad() 
        out_p_raw, out_p_masked, out_p_mask, out_p_ids_restore  = model_p(x) # (_, 1+196,768)、(_, 1+196-M,768)
        pred_p_raw = classifier_p(out_p_raw[:,0]) 
        out_s_raw, out_s_masked, out_s_mask, out_s_ids_restore = model_s(x) # (_, 1+196,768)、(_, 1+196-M,768)
        pred_s_raw = classifier_s(out_s_raw[:,0])
        
        # 进行重建，对称的
        mae_p = mae_decoder(out_p_masked, out_p_ids_restore) # # 返回所有的patch嵌入(-,196,768)，不包括class token
       
        # mask : torch.Size([-, 196]) 用于计算mae重建loss
        # ids_restore : (-, 196)
        
        
       
        # 下面计算ce loss
        loss_ce_p = loss_fn(pred_p_raw, y)
        loss_ce_s = loss_fn(pred_s_raw, y)
        
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
        loss = loss_ce_p + loss_ce_s + loss_p * args.lambda_mae_loss + loss_ss * args.lambda_ss_loss
        # 计算分类精度
        _, predicted = torch.max(pred_p_raw.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  
        # 更新参数
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        
    avg_loss = total_loss / float(i+1)
    acc = top1.avg
            
    return avg_loss, acc

def test(test_loader, model_p, classifier_p):
    top1 = utils.AverageMeter()
    model_p.eval()
    classifier_p.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            out_p_raw, out_p_masked, _, _ = model_p(x) # (_, 768)
            pred = classifier_p(out_p_raw[:,0]) 
           
            # 计算分类精度
            _, predicted = torch.max(pred.data, 1)
            correct = predicted.eq(y.data).cpu().sum()
            top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

    acc = top1.avg
            
    return acc

if __name__ == "__main__":
    
    # 定义接口
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_weights', default='pretrain_model.pth', type=str)
    parser.add_argument("--output-dir", default="", type=str, help="Output directory to write results and logs")
    parser.add_argument("--opts", help="Extra configuration options", default=[], nargs="+")
    parser.add_argument('--data_root', default='./dataset/images', type=str)
    parser.add_argument('--train_txt_path', default='./train.txt', type=str)
    parser.add_argument('--val_txt_path', default='./val.txt', type=str)
    parser.add_argument('--test_txt_path', default='./test.txt', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--shuffle', default='True', type=bool)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--seed' , default=1, type=int, help='seed')
    parser.add_argument('--save_dir', default='./logs', help='Save dir')
    parser.add_argument('--model_path', default='./logs', help='model_path')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='mask_ratio')
    parser.add_argument('--lambda_mae_loss', default=1.0, type=float, help='lambda_mae_loss')
    parser.add_argument('--lambda_ss_loss', default=1.0, type=float, help='lambda_ss_loss')
    
    
    
    
    args = parser.parse_args()
    
    # 固定随机性
    setup_seed(args.seed)
    
    # 保持模型的文件夹
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 验证数据
    val_loader = load_data(data_root=args.data_root, txt_path=args.val_txt_path, phase='val', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
    # 测试数据
    test_loader = load_data(data_root=args.data_root, txt_path=args.test_txt_path, phase='test', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
    
    

    
    
    # run 5次
    acc = []
    for i in range(5):
        # 训练数据
        train_txt_path = args.train_txt_path
        train_txt_path = train_txt_path.replace("shot", 'shot_run' + str(i+1))
        train_loader = load_data(data_root=args.data_root, txt_path=train_txt_path, phase='train', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
        # 定义pretrain模型
        model_type = "dino"
        model_name = "vit_base"
        model_patchsize = 16
        model_p = vits.__dict__[model_name](mask_ratio=args.mask_ratio, patch_size=model_patchsize, num_classes=0)
        model_p.cuda()
        state_dict = torch.load(args.pretrained_weights)
        model_p.load_state_dict(state_dict, strict=True)
        print('model_type:', model_type)
        print(f"Model {model_name} {model_patchsize}x{model_patchsize} built.") 
        # 定义分类器
        classifier_p = LinearClassifier(out_dim=768, num_classes=args.num_class).cuda()
        
        # 定义scrach模型
        model_type = "dino"
        model_name = "vit_base"
        model_patchsize = 16
        model_s = vits.__dict__[model_name](mask_ratio=args.mask_ratio, patch_size=model_patchsize, num_classes=0)
        model_s.cuda()
        print('model_type:', model_type)
        print(f"Model {model_name} {model_patchsize}x{model_patchsize} built.") 
        # 定义分类器
        classifier_s = LinearClassifier(out_dim=768, num_classes=args.num_class).cuda()
        
        # 定义mae decoder，共享参数的。
        mae_decoder = MAE_decoder.MAE_decoder(patch_size=model_patchsize, decoder_embed_dim=768).cuda()

        
        # 定义优化器
        optimizer = torch.optim.SGD([{"params":mae_decoder.parameters()}, {"params":model_p.parameters()}, {"params":model_s.parameters()}, {"params":classifier_p.parameters()}, {"params":classifier_s.parameters()}], lr=args.lr, momentum=0.9, weight_decay=0)
        loss_fn = nn.CrossEntropyLoss().cuda()
        
        # 定义训练接口
        best_val_acc = 0
        for epoch in range(args.epoch):
            train_loss, train_acc = train(train_loader, model_p, classifier_p, model_s, classifier_s, mae_decoder, loss_fn, optimizer, args)
            print('train:', epoch+1, 'current epoch train loss:', train_loss, 'current epoch train acc:', train_acc)
            if epoch+1 >= 100:
                val_acc = test(val_loader, model_p, classifier_p)
                print('val:', epoch+1, 'current epoch val acc:', val_acc)
                if val_acc > best_val_acc:
                    print('best epoch:', epoch+1)
                    best_val_acc = val_acc
                    outfile = os.path.join(args.save_dir, 'best.tar')
                    torch.save({'epoch':epoch+1, 'state_model_p':model_p.state_dict(), 'state_classifier_p':classifier_p.state_dict()}, outfile) # 只保存model的参数。
                
        # 所有epoch训练完之后，根据验证集选出来的模型进行测试
        tmp = torch.load(args.model_path)
        state_dict_model_p = tmp['state_model_p']
        model_p.load_state_dict(state_dict_model_p)
        
        state_dict_classifier_p = tmp['state_classifier_p']
        classifier_p.load_state_dict(state_dict_classifier_p)


            
        test_acc = test(test_loader, model_p, classifier_p)
        acc.append(test_acc)
        print('run:', i+1, 'test acc:', test_acc)
    print(acc)
    # 计算均值和方差
    acc_mean = np.mean(acc)
    #求方差
    acc_var = np.var(acc)
    print('mean:', acc_mean, 'var', acc_var)
    
        
        
    
        
    
