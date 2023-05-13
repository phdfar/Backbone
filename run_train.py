import random
import dataloader
import net

def start(args):
    if args.task=='correct':
        dataloader_train,dataloader_val=dataloader.correct.run(args)
        if args.network=='simple_backbone':
            net.simple_backbone.run(args,dataloader_train,dataloader_val)
        
        

  