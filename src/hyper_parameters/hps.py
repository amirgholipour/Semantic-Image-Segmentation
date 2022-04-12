import os
def get_hyper_paras():

    BATCH = 64
    STEPS_PER_EPOCH = 800//BATCH
    VALIDATION_STEPS = 200//BATCH
    EPOCHS = 50
    VAL_SUBSPLITS = 5
    FINE_TUNE = False

    base, sourceRepoName = os.path.split(os.getcwd())
    base, sourceRepoName = os.path.split(base)
    refRepoName = sourceRepoName.replace('Workshop','Inference')
    model_dir= base +'/'+refRepoName +'/'+ 'models/model.h5'
    

    refRepoDir = base +'/'+refRepoName 
    sourceRepoDir = base +'/'+sourceRepoName
    return BATCH,STEPS_PER_EPOCH,VALIDATION_STEPS,EPOCHS,VAL_SUBSPLITS,FINE_TUNE,model_dir,refRepoName,sourceRepoName,refRepoDir,sourceRepoDir