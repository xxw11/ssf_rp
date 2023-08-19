import torch.nn as nn
def adjust_all_parameters(args, model_or_params):
    tuning_mode = args.tuning_mode
    print("adjust_all_parameters:")
    if isinstance(model_or_params, nn.Module):
        if tuning_mode:
            for name, param in model_or_params.named_parameters():
                param.requires_grad = True
                if tuning_mode == 'ssf':
                    if  "ssf" in name: 
                        param.requires_grad = False
                elif tuning_mode == 'cross_ssf':
                    if  "ssf" in name: 
                        param.requires_grad = False
                if param.requires_grad == True:
                    print(name)
                

def adjust_partial_parameters(args, model_or_params):
    print("adjust_partial_parameters:")
    tuning_mode = args.tuning_mode
    if isinstance(model_or_params, nn.Module):
        if tuning_mode:
            for name, param in model_or_params.named_parameters():
                param.requires_grad = True
                if tuning_mode == 'linear_probe':
                    if "head." not in name:
                        param.requires_grad = False
                elif tuning_mode == 'ssf':
                    if "head." not in name and "ssf" not in name: 
                        param.requires_grad = False
                elif tuning_mode == 'tail_mlp':
                    if "tail_mlp" not in name: 
                        param.requires_grad = False
                elif tuning_mode == 'cross_ssf':
                    if "head." not in name and "ssf" not in name: 
                        param.requires_grad = False
                if param.requires_grad == True:
                    print(name)
                