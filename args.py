import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='RMSIN training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--bert_tokenizer', default='./bert_weight', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='./bert_weight', help='pre-trained BERT weights')
    parser.add_argument('--dataset', default='rrsisd', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=1024, type=int, help='input image size')
    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0,help='node rank for distributed training')
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_bifit', default=1e-4, type=float)
    parser.add_argument('--lr_3', default=1e-4, type=float)
    parser.add_argument('--lr_bifit_names',
                        default=['vlfusion_module','reins','vtfusion_module','feat_ifi','text_linear','query_ifi','mask_linear','mask_product','vmtoken','ifi_layers','sen_ifi','mtfusion_module','mask_text_fuser','mask_dsp','mask_iou_conv','mask_iou_linear','mmfusion_module','objfusion_module','mem_inter','fmfusion_module','mmfusion_layers','matextfusion_module','objfusion_layers','vtlfusion_module','vtlatten','MLP_Adapter','Space_Adapter','highlinear1','highlinear2','multi_ifi1','multi_ifi2','multi_ifi3','adapter','text_linear4','vlfusion_module4','text_linear3','vlfusion_module3','text_linear2','vlfusion_module2','text_linear1','vlfusion_module1','text_in_linear','local_conv','mlp_local','mlp_global','fc_fusion','lvfusion_module','text_out_linear'], type=str, nargs='+')
    parser.add_argument('--lr_3_names',
                        default=['vltoken_small','vltoken_large','vltoken','conv_small_re','conv_large_re','conv_small','conv_large','weight_mlp_s','weight_mlp','weight_mlp_l'], type=str, nargs='+')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='RMSIN', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='./swin_base_patch4_window12_384_22k.pth',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='./your_data/RRSIS-D', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--vision_pretrained',
                        default="./your_weights/sam2-hiera-large/sam2_hiera_large.pt",
                        type=str)  # sam
    # parser.add_argument('--vision_yaml', default="/horizon-bucket/saturn_v_dev/01_users/fu.rong/RVOS_evf/weights/sam2-hiera-large/sam2_hiera_l.yaml",type=str)
    parser.add_argument('--encoder_pretrained',
                        default="./your_weights/beit3_large_patch16_224.pth",
                        type=str)  # beit
    parser.add_argument('--train_mask_decoder', action='store_true', default=True)
    parser.add_argument('--train_prompt_encoder', action='store_true', default=True)
    parser.add_argument('--version', default="./your_weights/beit3.spm",
                        type=str)  # all
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--sam_scale", default='huge', type=str)
    parser.add_argument("--mm_extractor_scale", default='large', type=str)
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument('--inference', action='store_true', default=False)
    parser.add_argument('--get_pretrain', action='store_true', default=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
