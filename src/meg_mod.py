from pathlib import Path
import os
#import tensorflow as tf
from m3gnet.trainers import PotentialTrainer
from m3gnet.models import M3GNet, M3GNetCalculator, BasePotential, Potential, Relaxer
import pickle
import shutil

def add_data(tr, te, mo):
    with open(mo['directory_FN']+'/train.pkl', 'wb') as file:
        pickle.dump(tr, file)
    with open(mo['directory_FN']+'/test.pkl', 'wb') as file:
        pickle.dump(te, file)

def modify_m3gnet(empty_m3gnet_FN, m3gnet_options):
    mo= m3gnet_options
    with open(empty_m3gnet_FN, "r") as in_file:
        buf = in_file.read()

    for key, value in mo.items():
        if key.startswith('-'):
            buf = buf.replace(key[1:]+'!VAR',key[1:]+'='+value)
        if key.startswith('+'):
            buf = buf.replace(key[1:]+'!VAR',key[1:]+'=\"'+value+'\"')

    print(buf)
    if mo['--job-name'] != '':
        with open(mo['directory_FN']+'/'+mo['--job-name']+'.py', "w") as out_file:
            out_file.write(buf)


def create_SLURM_dir(empty_slurm_FN,dir_options,delete=False):
    '''function to create a SLURM directory and a slurm file where
    a SLURM_job can be created and nested to it, reference to this
    job will be by path... and a conditional for the pkl or json
    existence of this job will be exploited'''

    #from creat_slurm_job

    do=dir_options

    with open(empty_slurm_FN, "r") as in_file:
        buf = in_file.read()


    if len(do['spec_modules'])>0:
        buf=buf.replace('MODULE LOAD', 'module load '+ ' '.join(do['spec_modules']))

    if len(do['spec_path'])>0:
        dosp=[]
        for i in do['spec_path']:
            dosp.append('\"'+i+':$PATH'+'\"')
            buf=buf.replace('PATH LOAD', '\nexport PATH= '+ '\nexport PATH= '.join(dosp))
    else:
        buf=buf.replace('PATH LOAD', '\n')

    if len(do['spec_env_cmd'])>0:
        buf=buf.replace('ENV LOAD', do['spec_env_cmd'])
    else:
        buf=buf.replace('ENV LOAD', '')

    buf=buf.split('\n')

    #### command sen
    for i in range(len(buf)):
        if buf[i] == '#---------------------- job ----------------------------#':
            buf[i] = buf[i] + '\n' + do['env']+'/bin/'+do['spec_cmd_beg'] +' '+ do['--job-name']+'.py' + '\n'
    for i,str_i in enumerate(buf):
        for key, value in do.items():
            if '--' in key:
                if key in str_i:
                    buf[i]='#SBATCH '+key+'='+value


    buf=[i.split('\n') for i in buf]
    buf=[item for sublist in buf for item in sublist]

    buf='\n'.join(buf)

    folder_or_delete(do['directory_FN'],delete=delete)

    if do['--job-name'] != '':
        with open(do['directory_FN']+'/'+do['--job-name']+'.slurm', "w") as out_file:
            out_file.write(buf)
    print(buf)
    return None

def folder_or_delete(dir_name, delete=True):
    '''checks if there's a floder with that name, if not it creates it,
    if yes it deletes it and it creates it again'''

    if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    else:
        if delete:
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)
        else:
            pass

class m3g_case:
    def __init__(self,IDX):
        self.hists=[]
        self.run_dir_IDs=[]
        self.trains=[]
        self.tests=[]
        self.IDX = IDX
        
        if Path(self.IDX).exists():

            with open(self.IDX+'/'+'obj.pkl', 'rb') as file:
                #one=pickle.load(file).__dict__
                one=pickle.load(file)
                for key in one:
                    if not key.startswith('__'):
                        setattr(self, key, one[key])
            self.m3gnet_=M3GNet().from_dir(self.IDX+'/'+'m3gnet_curr')
        else:
            folder_or_delete(self.IDX)
            
            self.m3gnet_ = M3GNet(is_intensive=False).load()
            self.m3gnet_.save(self.IDX+'/'+'m3gnet_curr')
            self.save_pkl()
        
    def start_dir(self, IDX):
        self.IDX = IDX
        if Path(self.IDX).exists():
            pass
        else:
            self.m3gnet_ = M3GNet(is_intensive=False).load()
            self.m3gnet_.save(self.IDX+'/'+'m3gnet_curr')
            
    def prepare_for_run(self, dic, re_data=False, delete=True):
        
        rdir = dic['directory_FN']
        folder_or_delete(rdir,delete)
        self.m3gnet_.save(rdir+'/m3gnet_start')
        
        create_SLURM_dir('/scratch/user/guillermo.vazquez/SAVE/EMPTY_FILES/runEMPTY-mod.slurm',dic)
        modify_m3gnet('/scratch/user/guillermo.vazquez/SAVE/EMPTY_FILES/megnet_xxxx.py', dic)
        
        if re_data == True:
            return rdir,self.trains[-1],self.tests[-1]
        else:
            return rdir
        
    def update_model_from_dict(self,dic, slurm, remove_dir=False):
        rdir=dic['directory_FN']
        
        if Path(rdir+'/'+'m3gnet_end').exists():
            self.m3gnet_=M3GNet().from_dir(rdir+'/'+'m3gnet_end')
            self.m3gnet_.save(self.IDX+'/'+'m3gnet_curr')
        
        
        
        work_id = slurm.ID
        if work_id in self.run_dir_IDs:
            self.update(dic, soft='on')
            
        else:
            self.hists.append(0)
            self.update(dic,soft='off')
            self.run_dir_IDs.append(work_id)
            
        #slurm.squeue_check()
        #return slurm.status
        
    def update(self,dic, soft='on'):
        
                
        if soft=='off':        
            if Path(dic['directory_FN']+'/'+'train.pkl').exists():
                with open(dic['directory_FN']+'/'+'train.pkl', 'rb') as file:
                    tr_it = pickle.load(file)
                    self.trains.append(tr_it)
            if Path(dic['directory_FN']+'/'+'test.pkl').exists():
                with open(dic['directory_FN']+'/'+'test.pkl', 'rb') as file:
                    te_it = pickle.load(file)
                    self.tests.append(te_it)
                    
                    
        if Path(dic['directory_FN']+'/'+'his_train.pkl').exists():
            with open(dic['directory_FN']+'/'+'his_train.pkl', 'rb') as file:
                hist_it = pickle.load(file)
                self.hists[-1]=hist_it.copy()
                    
    def __repr__ (self):
        string_repr = ''
        for key, value in self.__dict__.items():
            string_repr= string_repr + str(key)+'    :\n'+ str(value) + '\n'
        return string_repr
    

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['m3gnet_']
        return d
        
    def save_pkl(self):
        with open(self.IDX+'/'+'obj.pkl', 'wb') as file:
            d = dict(self.__dict__)
            del d['m3gnet_']
            pickle.dump(d, file)
