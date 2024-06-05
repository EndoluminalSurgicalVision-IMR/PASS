######################
######RIGA#####
######################
Base1_config = {'num_classes':2, 'root':'data/RIGAPlus/', 'pretrained_model':'log_dir/UNet_Source_RIGA/checkpoints/model_best.model', 'tr_csv':['data/RIGAPlus/MESSIDOR_Base1_test.csv'], 'ts_csv':['data/RIGAPlus/MESSIDOR_Base1_test.csv'], 'patch_size': (512, 512), 'tag':'Base1_test'}


Base2_config = {'num_classes':2, 'root':'data/RIGAPlus/', 'pretrained_model':'log_dir/UNet_SourceRIGA/checkpoints/model_best.model', 'tr_csv':['data/RIGAPlus/MESSIDOR_Base2_test.csv'], 'ts_csv':['data/RIGAPlus/MESSIDOR_Base2_test.csv'],'patch_size': (512, 512), 'tag':'Base2_test'}


Base3_config = {'num_classes':2, 'root':'RIGAPlus/', 'pretrained_model':'log_dir/UNet_Source_RIGA/checkpoints/model_best.model', 'tr_csv':['RIGAPlus/MESSIDOR_Base3_test.csv'], 'ts_csv':['RIGAPlus/MESSIDOR_Base3_test.csv'], 'patch_size': (512, 512), 'tag':'Base3_test'}

Base1_tent_config = {'arch': 'unet_2d', 'num_classes':2, 'batch_size': 8, 'continue_training': False, 'gpu': [0], 'initial_lr': 0.001, 'log_folder': 'log_dir', 'manualseed': 100, 'model_episodic': False, 'num_threads': 0, 'optim_steps': 2, 'optimizer': 'Adam', 'patch_size': [512, 512], 'pretrained_model': 'log_dir/UNet_Source_RIGA/checkpoints/model_best.model',  'root': 'data/RIGAPlus/', 'tag': 'Base1_test', 'ts_csv': ['data/RIGAPlus/MESSIDOR_Base1_test.csv']}

Base2_tent_config = {'arch': 'unet_2d', 'num_classes':2, 'batch_size': 8, 'continue_training': False, 'gpu': [0], 'initial_lr': 0.001, 'log_folder': 'log_dir', 'manualseed': 100, 'model_episodic': False, 'num_threads': 0, 'optim_steps': 2, 'optimizer': 'Adam', 'patch_size': [512, 512], 'pretrained_model': 'log_dir/UNet_Source_RIGA/checkpoints/model_best.model', 'root': 'data/RIGAPlus/', 'tag': 'Base2_test', 'ts_csv': ['data/RIGAPlus/MESSIDOR_Base2_test.csv']}

Base3_tent_config = {'arch': 'unet_2d', 'num_classes':2, 'batch_size': 8, 'continue_training': False, 'gpu': [0], 'initial_lr': 0.001, 'log_folder': 'log_dir', 'manualseed': 100, 'model_episodic': False, 'num_threads': 0, 'optim_steps': 2, 'optimizer': 'Adam', 'patch_size': [512, 512], 'pretrained_model': 'log_dir/UNet_Source_RIGA/checkpoints/model_best.model',  'root': 'data/RIGAPlus/', 'tag': 'Base3_test', 'ts_csv': ['data/RIGAPlus/MESSIDOR_Base3_test.csv']}


######################
###### MRI-Prostate####
## ####################


Prostate_RUNMC2BMC_config = {'num_classes':1, 'root':'../Medical_TTA/data/MRI_prostate', 'model_episodic':False, 'pretrained_model':'log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model', 'tr_csv':['../Medical_TTA/data/MRI_prostate/BMC_all.csv'], 'ts_csv':['../Medical_TTA/data/MRI_prostate/BMC_all.csv'], 'patch_size': (384, 384), 'tag':'Prostate_RUNMC2BMC', 'batch_size':16, 'loss_dict':['bn_loss',{'layers':9, 'alpha':0.01}]}


Prostate_RUNMC2BIDMC_config = {'num_classes':1, 'root':'../Medical_TTA/data/MRI_prostate', 'model_episodic':True, 'pretrained_model':'log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model', 'tr_csv':['../Medical_TTA/data/MRI_prostate/BIDMC_all.csv'], 'ts_csv':['../Medical_TTA/data/MRI_prostate/BIDMC_all.csv'], 'patch_size': (384, 384), 'tag':'Prostate_RUNMC2BIDMC', 'batch_size':16, 'loss_dict':['bn_loss',{'layers':9, 'alpha':0.01}]}


Prostate_RUNMC2HK_config = {'num_classes':1, 'root':'../Medical_TTA/data/MRI_prostate', 'model_episodic':False, 'pretrained_model':'log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model', 'tr_csv':['../Medical_TTA/data/MRI_prostate/HK_all.csv'], 'ts_csv':['../Medical_TTA/data/MRI_prostate/HK_all.csv'], 'patch_size': (384, 384), 'tag':'Prostate_RUNMC2HK', 'batch_size':16, 'loss_dict':['bn_loss',{'layers':9, 'alpha':0.01}]}


Prostate_RUNMC2UCL_config = {'num_classes':1, 'root':'../Medical_TTA/data/MRI_prostate', 'model_episodic':False,'pretrained_model':'log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model', 'tr_csv':['../Medical_TTA/data/MRI_prostate/UCL_all.csv'], 'ts_csv':['../Medical_TTA/data/MRI_prostate/UCL_all.csv'], 'patch_size': (384, 384), 'tag':'Prostate_RUNMC2UCL', 'batch_size':16, 'loss_dict':['bn_loss',{'layers':9, 'alpha':0.01}]}

Prostate_RUNMC2I2CVB_config = {'num_classes':1, 'root':'../Medical_TTA/data/MRI_prostate', 'model_episodic':False, 'pretrained_model':'log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model', 'tr_csv':['../Medical_TTA/data/MRI_prostate/I2CVB_all.csv'], 'ts_csv':['../Medical_TTA/data/MRI_prostate/I2CVB_all.csv'], 'patch_size': (384, 384), 'tag':'Prostate_RUNMC2I2CVB', 'batch_size':16, 'loss_dict':['bn_loss',{'layers':9, 'alpha':0.01}]}

