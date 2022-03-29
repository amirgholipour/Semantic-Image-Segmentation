import os
def get_hyper_paras():
    BATCH = 64
    STEPS_PER_EPOCH = 800//BATCH
    VALIDATION_STEPS = 200//BATCH
    EPOCHS = 100
    VAL_SUBSPLITS = 5
    FINE_TUNE = False
    base, sourceRepoName = os.path.split(os.getcwd())
    refRepoName = 'SIS-Inference'
    model_dir= base +'/'+refRepoName +'/'+ 'models/SemImSeg_model_EfficientNetV2B0_.h5'
    
    refRepoDir = base +'/'+refRepoName 
    return BATCH,STEPS_PER_EPOCH,VALIDATION_STEPS,EPOCHS,VAL_SUBSPLITS,FINE_TUNE,model_dir,refRepoName,sourceRepoName,refRepoDir