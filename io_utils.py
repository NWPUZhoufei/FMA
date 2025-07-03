import argparse


def parse_args_test():
    parser = argparse.ArgumentParser(description='test')
    # 数据层参数配置
    parser.add_argument('--EuroSAT_data_path', default='/data1/zhoufei/CDFSL/datasets/EuroSAT/2750', help='EuroSAT_data_path') # 数据路径
    parser.add_argument('--mini_test_data_path', default='/data0/zhoufei/CDFSL/datasets/miniImagenet/test', help='mini_test_data_path') # 数据路径
    parser.add_argument('--mini_test_class' , default=20, type=int, help='total number of classes in EuroSAT') 
    parser.add_argument('--image_size'  , default=224, type=int,  help='image size') # 输入图片大小
    parser.add_argument('--EuroSAT_class' , default=10, type=int, help='total number of classes in EuroSAT') 
    parser.add_argument('--feature_size' , default=512, type=int, help='feature_size')
    # backbone的配置，可以改变每个block输出的通道个数以及整个backbone输出的feature map大小
    parser.add_argument('--list_of_out_dims', default=[64,128,256,512], help='every block output')
    parser.add_argument('--list_of_stride', default=[1,2,2,2], help='every block conv stride')
    parser.add_argument('--list_of_dilated_rate', default=[1,1,1,1], help='dilated conv') # 是否使用空洞卷积
    # 模型保存相关参数配置
    parser.add_argument('--model_path', default='./log/best_model.tar', help='model_path')
    # eposide采样配置
    parser.add_argument('--n_way', default=5, type=int,  help='class num to classify for every task')
    parser.add_argument('--n_support', default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--n_query', default=15, type=int,  help='number of test data in each class, same as n_query') 
    parser.add_argument('--test_n_eposide', default=600, type=int, help ='total task every epoch') # for meta-learning methods, each epoch contains 100 episodes
    parser.add_argument('--seed' , default=1111, type=int, help='feature_size')
    
    parser.add_argument('--current_data_path', default='/data0/zhoufei/CDFSL/datasets/ISIC', help='ISIC_data_path') # 数据路径
    parser.add_argument('--current_class', default=7, type=int, help='total number of classes in ISIC')


    parser.add_argument('--model_path1', default='./log/best_model.tar', help='model_path1')
    parser.add_argument('--model_path2', default='./log/best_model.tar', help='model_path2')
    parser.add_argument('--model_path3', default='./log/best_model.tar', help='model_path3')
    parser.add_argument('--model_path4', default='./log/best_model.tar', help='model_path4')
    parser.add_argument('--model_path5', default='./log/best_model.tar', help='model_path5')
    parser.add_argument('--model_path6', default='./log/best_model.tar', help='model_path6')
    parser.add_argument('--model_path7', default='./log/best_model.tar', help='model_path7')
    parser.add_argument('--model_path8', default='./log/best_model.tar', help='model_path8')
    parser.add_argument('--model_path9', default='./log/best_model.tar', help='model_path9')
    parser.add_argument('--model_path10', default='./log/best_model.tar', help='model_path10')
    parser.add_argument('--model_path11', default='./log/best_model.tar', help='model_path11')
    parser.add_argument('--model_path12', default='./log/best_model.tar', help='model_path12')
    parser.add_argument('--model_path13', default='./log/best_model.tar', help='model_path13')
    parser.add_argument('--model_path14', default='./log/best_model.tar', help='model_path14')
    parser.add_argument('--model_path15', default='./log/best_model.tar', help='model_path15')
    parser.add_argument('--model_path16', default='./log/best_model.tar', help='model_path16')
    parser.add_argument('--model_path17', default='./log/best_model.tar', help='model_path17')
    parser.add_argument('--model_path18', default='./log/best_model.tar', help='model_path18')
    parser.add_argument('--model_path19', default='./log/best_model.tar', help='model_path19')
    parser.add_argument('--model_path20', default='./log/best_model.tar', help='model_path20')
    parser.add_argument('--model_path21', default='./log/best_model.tar', help='model_path21')
    parser.add_argument('--model_path22', default='./log/best_model.tar', help='model_path22')
    parser.add_argument('--model_path23', default='./log/best_model.tar', help='model_path23')
    parser.add_argument('--model_path24', default='./log/best_model.tar', help='model_path24')

    
    
    parser.add_argument('--DS_N', default=750, type=int,  help='number of gen data in each class') 
    
    parser.add_argument('--n_aug_support_samples', default=5, type=int, help='total number of n_aug_support_samples')
    parser.add_argument('--ft_lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--pseudo_n_query', default=5, type=int,  help='number of test data in each class, same as pseudo n_query') 
    parser.add_argument('--pseudo_total_epoch', default=100, type=int, help='initial learning rate')
    parser.add_argument('--topK' , default=10, type=int, help='topK')
    parser.add_argument('--topK1' , default=200, type=int, help='topK1')
    parser.add_argument('--topK2' , default=100, type=int, help='topK2')
    parser.add_argument('--crop_num' , default=3, type=int, help='crop_num')
    parser.add_argument('--kmeans_niter' , default=10, type=int, help='kmeans_niter')  
    parser.add_argument('--lamba1', type=float, default=1.0, help='lamba_cross') 
    parser.add_argument('--crop_size' , default=96, type=int, help='crop_size')
    parser.add_argument('--patch_size' , default=112, type=int, help='patch_size')
    
    parser.add_argument('--tr_N' , default=7, type=int, help='tr_N')
    parser.add_argument('--tr_K' , default=10, type=int, help='tr_K')
    
    parser.add_argument('--base_means_path', default='./dc_feature/miniImageNet_base_means.npy', help='base_means_path')
    parser.add_argument('--base_cov_path', default='./dc_feature/miniImageNet_base_cov.npy', help='base_cov_path')
    parser.add_argument('--clip_model_path', default='./clip/ ', help='clip_model_path')
    
    # dino
    parser.add_argument('--dino_model_path', default='dino_deitsmall8_pretrain.pth', help='dino_model_path')
    parser.add_argument('--dino_arch', default='vit_small', help='dino_arch')
    parser.add_argument('--dino_patch_size' , default=16, type=int, help='dino_patch_size')
    
    parser.add_argument('--dataset_type', default='datasetname', help='dataset_type')
    
    
    
    parser.add_argument('--sample_number' , default=50, type=int, help='sample_number')
    parser.add_argument('--ju_alpha', default=1.0, type=float, help='ju_alpha')
    parser.add_argument('--gama', default=1.0, type=float, help='gama')
    parser.add_argument('--beta', default=0.5, type=float, help='beta')
    
    parser.add_argument('--data_path', default='/data1/zhoufei/CDFSL/datasets/miniImagenet', help='train data path') # 数据路径
    
    parser.add_argument('--data_name', default='ISIC', help='data_name')
    parser.add_argument('--model_name', default='dino', help='model_name')
    parser.add_argument('--num_class' , default=7, type=int, help='num_class')
    parser.add_argument('--sup_shot' , default=5, type=int, help='sup_shot')
    parser.add_argument('--unsup_shot' , default=10, type=int, help='unsup_shot')
    parser.add_argument('--test_shot' , default=100, type=int, help='test_shot')
    
    parser.add_argument('--deepemd', default='fcn', help='deepemd')
    parser.add_argument('--deepemd_model_dir', default='./deepEMD_model/max_acc.pth', help='deepemd_model_dir')
    parser.add_argument('-feature_pyramid', type=str, default=None)
    # solver
    parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
    # SFC
    parser.add_argument('-sfc_lr', type=float, default=100)
    parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('-sfc_update_step', type=float, default=100)
    parser.add_argument('-sfc_bs', type=int, default=4)
    parser.add_argument('-norm', type=str, default='center', choices=[ 'center'])
    parser.add_argument('-metric', type=str, default='cosine', choices=[ 'cosine' ])
    parser.add_argument('-temperature', type=float, default=12.5)
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    
    parser.add_argument('--save_name', type=str, default='source_data.npy')
    parser.add_argument('--image_path', type=str, default='source_data.npy')
    parser.add_argument('--save_path', type=str, default='source_data.npy')

    parser.add_argument('--gen_number' , default=10, type=int, help='gen_number')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='mask_ratio')
    parser.add_argument('--lambda_mae_loss', default=1.0, type=float, help='lambda_mae_loss') # 传统cdfsl设置下，默认损失系数设置为1
    parser.add_argument('--lambda_ss_loss', default=1.0, type=float, help='lambda_ss_loss')# 传统cdfsl设置下，默认损失系数设置为1
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    
    parser.add_argument('--total_epoch' , default=100, type=int, help='total_epoch')
    

    return parser.parse_args()










