import random
from dataloader import correct,binary
from net import simple_backbone,simple_binary

def start(args):
    if args.task=='correct':
        dataloader_train,dataloader_val=correct.run(args)
        if args.network=='simple_backbone':
            simple_backbone.run(args,dataloader_train,dataloader_val)
    
    elif args.task=='binary':
        dataloader_train,dataloader_val=binary.run(args)
        if args.network=='simple_binary':
            simple_binary.run(args,dataloader_train,dataloader_val)
        
        

  