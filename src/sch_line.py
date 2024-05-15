import os
import pickle
import numpy as np
import shutil
import copy

from pyiron import Project
from pyiron import ase_to_pyiron


from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io import ase as pgase

ase_to_pmg = pgase.AseAtomsAdaptor.get_structure
pmg_to_ase = pgase.AseAtomsAdaptor.get_atoms







class pyiron_line:
    
    def __init__(self, 
                 project_pyiron: str = 'testing_cons',
                 **kwargs):
        
        self.project_STR = project_pyiron
        if os.path.exists('{}.pkl'.format(self.project_STR)):
            self.load_obj()
            self.PRE_NEEDED = False
            return None
        else:
            self.PRE_NEEDED = True
        
        self.dft_rores_HIS = []
        self.dft_rooster_HIS = []
        self.cur_rooster_HIS = []
        self.steps_HIS = []
        
        if 'NUM_cores' in kwargs:
            self.NUM_cores = kwargs['NUM_cores']
        else:
            self.NUM_cores = 20

        if 'extra' in kwargs:
            self.extra = kwargs['extra']

        if 'vasp_chosen' in kwargs:
            self.vasp_chosen = kwargs['vasp_chosen']
        else:
            self.vasp_chosen = 3
            
        if 'args_cons' in kwargs:
            self.args_cons = kwargs['args_cons']
        else:
            self.args_cons = {}
            
        if 'soft_STOP' in kwargs:
            self.soft_STOP = kwargs['soft_STOP']
        else:
            self.soft_STOP = 20
        
        if 'master_blank' in kwargs:
            self.master_blank = kwargs['master_blank']
        else:
            self.master_blank = {'structure': None,
                                 'job_specs': {
                                        '-INCAR-NCORE': 16,
                                        '-INCAR-LORBIT': 1,
                                        '-INCAR-NSW': 200,
                                        '-INCAR-PREC': 'Accurate',
                                        '-INCAR-IBRION': 2,
                                        '-INCAR-ISMEAR': 0,
                                        '-INCAR-SIGMA': 0.03,
                                        '-INCAR-ENCUT': 400,
                                        '-INCAR-EDIFF': 1E-6,
                                        '-INCAR-ISIF': 2,
                                        '-INCAR-ISYM': 0,
                                        'KPPA': 3000,
                                        '-INCAR-ISPIN': 1,
                                        }}

    """
    #### IMPORTANT handling of vasp command script
    #### choose vasp5 or vasp6
    vasps = ['vasp_5.4.4.default_grace',
             'vasp_6.3.0.default_grace',
             'vasp_6.3.2.default_grace',
             'vasp_6.3.2.default_faster',
             'vasp_6.3.2.default_faster2']
    RV_DIR = '/scratch/group/arroyave_lab/guillermo.vazquez/pyiron_dirs/pyiron/resources/vasp/bin'

    for filename in os.listdir(RV_DIR):
        f = os.path.join(RV_DIR, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f.endswith('.sh'):
                os.remove(f)
            


    vasp_opt=vasps[3]
    print('run_vasp'+vasp_opt.split('vasp')[1].split('default')[0]+'sh')
    shutil.copy2(RV_DIR+'/'+vasp_opt,RV_DIR+'/'+'run_vasp'+vasp_opt.split('vasp')[1].split('default')[0]+'sh')


    """
    def _vasp_HPRC_set(self):
        vasp_scripts = ['vasp_5.4.4.default_grace',
         'vasp_6.3.0.default_grace',
         'vasp_6.3.2.default_grace',
         'vasp_6.3.2.default_faster',
         'vasp_6.3.2.default_faster_VTST']
        RV_DIR = '/scratch/group/arroyave_lab/guillermo.vazquez/pyiron_dirs/pyiron/resources/vasp/bin'
        for filename in os.listdir(RV_DIR):
            f = os.path.join(RV_DIR, filename)
            # checking if there is an .sh file and deletes it
            if os.path.isfile(f):
                if f.endswith('.sh'):
                    os.remove(f)
        vasp_chosen = vasp_scripts[self.vasp_chosen]
        shutil.copy2(RV_DIR+'/'+vasp_chosen,
                     RV_DIR+'/'+'run_vasp' +
                     vasp_chosen.split('vasp')[1].split('default')[0]+'sh')


    def print_cut(self):
        print('#'*50+'\n')
        
    def save_obj(self):
        with open('{}.pkl'.format(self.project_STR), "wb") as fp:
            d = dict(self.__dict__)
            pickle.dump(d, fp)

    def load_obj(self):
        with open('{}.pkl'.format(self.project_STR), "rb") as fp:
            d = pickle.load(fp)
        self.__dict__.update(d)

    def pre_run(self, structure_ASES):
        if not self.PRE_NEEDED:
            print('pre_run() not needed')
            print('current stage: {}'.format(self.stage))
            print('current indices_RUN: {}'.format(self.indices_RUN))
            return
        self.stage = 0

        self.indices_RUN = np.arange(self.NUM_cores)
        self.job_crrnt = self.NUM_cores
        self.structure_ASES = structure_ASES
        self.cur_rooster = np.arange(len(structure_ASES))
        self.size_rooster = len(self.cur_rooster)
        self.dft_rooster = [[] for _ in range(self.size_rooster)]
        self.dft_rores = [[] for _ in range(self.size_rooster)]
        self.err_arr = self.create_errors()
        self.Xp = []
        self.yp = []
        self.RUNNING = True
        print('locked object into {}.pkl'.format(self.project_STR))
        self.save_obj()
    
    def update_rooster(self, fun):
        if len(self.dft_rores_HIS) == self.stage:
            
            self.Xi, self.yi, self.indices_i = self._assess_rooster()
            if len(self.yi) < 1:
                print('no new results have been added to pool')
            else:
                self.Xp.append(self.Xi)
                self.yp.append(self.yi)
                # min_ind = np.argmin(self.yp)

                self.steps_HIS.append(
                    [len(self.yp), 'actual iteration or gradient or idk',])
                self.dft_rores_HIS.append(self.dft_rores)
                self.dft_rooster_HIS.append(self.dft_rooster)
                self.cur_rooster_HIS.append(self.cur_rooster)
                
                self.stage = self.stage + 1
                
                self.indices_RUN = np.arange(self.NUM_cores)
                self.job_crrnt = self.NUM_cores
        
        new_x = fun(self.Xp, self.yp)
        print(new_x)
        for xj in new_x:
            print(_generate_struct_args(xj, self.full_lims))
        print(_get_euclidean(new_x))
        if np.any(np.abs(_get_euclidean(new_x)) < 1E-6):
            print('NEED TO RUN AGAIN')
            self.size_rooster = 0
            self.save_obj()
            return
        self.cur_rooster = new_x.copy()
        self.size_rooster = len(self.cur_rooster)
        self.dft_rooster = [[] for _ in range(self.size_rooster)]
        self.dft_rores = [[] for _ in range(self.size_rooster)]
        
        if self.stage > self.soft_STOP:
            print('stage larger than {}, proceeding to stop run'.format(
                self.soft_STOP))
            self.save_obj()
            return

        self.RUNNING = True
        self.save_obj()
    
    def run_step(self):
        
        if not self.RUNNING:
            # self.update_rooster(self.fun)
            return        
        self._vasp_HPRC_set()
        # run if RUNNING
        proj_pyiron = Project(path=self.project_STR)
        jobs = [0]*self.NUM_cores
        for i in range(self.NUM_cores):
            print('!'*50)
            print('core : ', i)
            print('!'*50 + '\n')
            
            it_in_rooster = self.indices_RUN[i]
            if it_in_rooster >= self.size_rooster:
                print('CORE has ended its use')
                continue
            print('rooster index: {}'.format(it_in_rooster))
            print(self.cur_rooster[it_in_rooster])
            it_vec = self.cur_rooster[it_in_rooster]
            
            it_strct = self.structure_ASES[it_vec]
            
            if len(self.dft_rooster[it_in_rooster]) == 0:

                master_b = copy.deepcopy(self.master_blank)
                if self.stage > 0:
                    if len(self.dft_rores_HIS[-1]) == 1:
                        it_strct.set_scaled_positions(
                            self.dft_rores_HIS[-1][0][0][
                                'trajectories'][-1].get_scaled_positions())
                    else:
                        it_strct.set_scaled_positions(
                                self.dft_rores_HIS[-2][0][0][
                                    'trajectories'][-1].get_scaled_positions())

                master_b['structure'] = it_strct
                self.dft_rooster[it_in_rooster] = copy.deepcopy(master_b)
            # print(self.dft_rooster)
            print(it_strct)
            
            try:
                jobs[i] = proj_pyiron.create_job(
                    job_type=proj_pyiron.job_type.Vasp,
                    job_name='vasp_{:03}'.format(
                                      self.stage)+'_{:03}'.format(
                                        it_in_rooster
                                      ),
                    delete_existing_job=False)
            except Exception as e:
                print('Exception in loading job: {}'.format(e))
                jobs[i] = proj_pyiron.load('vasp_{:03}'.format(
                                      self.stage)+'_{:03}'.format(
                                        it_in_rooster
                                      ))
            print(jobs[i].status)
            if jobs[i].status == 'initialized':
                jobs[i].structure = ase_to_pyiron(self.dft_rooster[
                    it_in_rooster]['structure'].copy())
                # job function change
                print('PASS to RUN --> rooster{:03}_strct{:03}'.format(
                        self.stage, it_in_rooster
                      ))
                
                _relax_blank(
                    jobs[i], self.dft_rooster[it_in_rooster]['job_specs'])
                continue

            if jobs[i].status == 'running' or \
                    jobs[i].status == 'collect':
                
                continue

            if jobs[i].status == 'finished' or jobs[i].status == 'aborted' \
                    or jobs[i].status == 'not_converged' or \
                    jobs[i].status == 'warning':
                try:

                    self._success(jobs[i], it_in_rooster)
                except Exception as e:
                    err_count = self._fail(jobs[i], it_in_rooster, e)
                    if err_count >= 5:
                        print('giving up on this run')
                    else:
                        self._fail_update(
                            jobs, i, it_in_rooster, proj_pyiron, err_count)
                        continue
                # finished worked
                self._next_it(i)
        print('end of run_step(),\ncurrent indices_RUN: {}'.format(
            self.indices_RUN))
        if np.all(self.indices_RUN >= self.size_rooster):
            self.RUNNING = False
        self.save_obj()
        
    def get_errs(self):
        directories = []
        inds = []
        directory = self.project_STR
        if os.path.isdir(directory):
            for entry in os.scandir(directory):
                if entry.is_dir():
                    if 'err' in entry.name:
                        directories.append(entry.name)

                        ind = entry.name.split('_')
                        ind = ind[1:3]
                        ind = [int(i) for i in ind]
                        inds.append(ind)
            if len(inds) > 0:
                return [np.array(inds), np.array([[-1, -1]]*len(inds)), 
                        directories, [{}]*len(inds)]
            else: 
                return [np.empty((0, 2), dtype=int), 
                        [], []]
        else:
            return [np.empty((0, 2), dtype=int), 
                    [], []]
        
    def create_errors(self):
        return [np.empty((0, 2), dtype=int), 
                np.empty((0, 2), dtype=int), [], []]
    
    def _success(self, job, it_in_r):
        self.dft_rores[
            it_in_r].append(
                self.get_dft_results_pyiron(job))
        print('pass {}'.format(job.status))
        self._zip_job_dir(job.name + '_hdf5')
        self.print_cut()
        
    def _fail(self, job, it_in_r, error_string):
        #  count errors to err_count
        
        err_count = np.count_nonzero(np.all(
            self.err_arr[0] == np.array([
                self.stage, 
                it_in_r]), axis=1))
        
        #  count errors to err_count
        self.err_arr[0] = \
            np.vstack((self.err_arr[0], np.array([
                self.stage,
                it_in_r])))
        
        self.err_arr[1] = \
            np.vstack((self.err_arr[1], np.array([
                -1,
                err_count])))

        self.err_arr[2].append(
            job.name+'_err_{:02}'.format(err_count))

        self.err_arr[3].append(copy.deepcopy(
            self.dft_rooster[it_in_r]))
        
        job.copy_to(new_job_name=job.name + '_err_{:02}'.format(err_count), 
                    copy_files=True, 
                    new_database_entry=False)
        print('error status: {} -o- error: {}'.format(
            job.status, error_string))
        self._zip_job_dir(job.name + '_err_{:02}'.format(err_count) + '_hdf5')
        self.print_cut()
        
        return err_count
    
    def _zip_job_dir(self, dir_name, DEL_AFTER=True):
        _zip_archive(self.project_STR + '/' + dir_name,
                     self.project_STR + '/' + dir_name + '.zip')
        if DEL_AFTER:
            shutil.rmtree(self.project_STR + '/' + dir_name)
    
    def _fail_update(self, job_list, core_i, it_in_r, proj_p, err_count):
        
        job_list[core_i] = proj_p.create_job(
            job_type=proj_p.job_type.Vasp,
            job_name='vasp_{:03}_{:03}'.format(
                    self.stage,
                    it_in_r
                          ),
            delete_existing_job=True)
        
        # start changing values after three tries of the same
        
        self.dft_rooster[
            it_in_r]['job_specs']['KPPA'] = \
            self.master_blank['job_specs'][
            'KPPA'] + 500*np.clip(err_count - 2, 0, 10)
        self.dft_rooster[
            it_in_r]['job_specs']['-INCAR-ENCUT'] = \
            self.master_blank['job_specs'][
            '-INCAR-ENCUT'] + 20*np.clip(err_count - 2, 0, 10) 
        
    def _next_it(self, core_i):
        self.indices_RUN[core_i] = self.job_crrnt
        self.job_crrnt = self.job_crrnt + 1
    
    def _assess_rooster(self):
        X = []
        y = []
        indices = []
        
        for i in range(self.size_rooster):
            
            if len(self.dft_rores[i]) > 0:
                if len(self.dft_rores[i][0]['energies']) > 1:
                    if self.dft_rores[i][0]['energies'][-1] < 0.0:
                        X.append(self.cur_rooster[i])
                        y.append(self.dft_rores[i][0]['energies'][-1])
                        indices.append([self.stage, i])
                    continue
                if self.dft_rores[i][0]['energies'][0] < 0.0:
                    X.append(self.cur_rooster[i])
                    y.append(self.dft_rores[i][0]['energies'][-1])
                    indices.append([self.stage, i])
        
        X, y = np.array(X), np.ravel(np.array(y))
        return X, y, indices
            
    def abort_specific(self, it_in_rooster, DELETE=False):
        
        proj_pyiron = Project(path=self.project_STR)
        try:
            job = proj_pyiron.create_job(
                job_type=proj_pyiron.job_type.Vasp,
                job_name='vasp_{:03}'.format(
                                  self.stage)+'_{:03}'.format(
                                    it_in_rooster
                                  ),
                delete_existing_job=False)
        except Exception as e:
            print('Exception in loading job: {}'.format(e))
            job = proj_pyiron.load('vasp_{:03}'.format(
                                  self.stage)+'_{:03}'.format(
                                    it_in_rooster
                                  ))
        
        print('status : ', job.status)
        
        job.drop_status_to_aborted()

        print('status : ', job.status)
        
    def get_dft_results_pyiron(self, job_func):
        return {'job_name': job_func.job_name,
                'status': job_func.status.string,
                'forces': job_func['output/generic/forces'].copy(),
                'stresses': job_func['output/generic/stresses'].copy(),
                'energies': job_func['output/generic/energy_pot'].copy(),
                'trajectories': [
                    j.to_ase().copy() for j in job_func.trajectory()],
                'internal_energies': job_func[
                    'output/generic/dft/scf_energy_int'].copy()}


def _zip_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, format), destination)
        

def _relax_blank(job, dic):

            
    
    
    for key, value in dic.items():
#         if key == '-INCAR-MAGMOM':
#             job.structure.set_initial_magnetic_moments(
#                 dic['-INCAR-MAGMOM'])
#             continue
        if key.startswith('-INCAR-'):
            job.input.incar[key.split('-INCAR-')[-1]] = value

            
    if '-INCAR-ISPIN' in dic.keys():
        if dic['-INCAR-ISPIN'] == 2:
            # if 'Fe',' Ni' in job.structure set ISPIN 1 and break
            obj_magmoms = CollinearMagneticStructureAnalyzer(ase_to_pmg(job.structure),
                                                            'replace_all')
            job.structure.set_initial_magnetic_moments(
                obj_magmoms.magmoms)
    
    if 'KPPA' in dic.keys():
        KPPA = dic['KPPA']
    else:
        KPPA = 3000

    if 'queue_number' in dic.keys():
        job.server.queue = job.server.list_queues()[dic['queue_number']]
    else:
        job.server.queue = job.server.list_queues()[0]

    if len(job.structure) == 1:
        job.server.queue = job.server.list_queues()[0]
        job.input.incar['NCORE'] = 1
    elif len(job.structure) > 1 and len(job.structure) <= 5:
        job.server.queue = job.server.list_queues()[1]
    elif len(job.structure) > 5 and len(job.structure) <= 10:
        job.server.queue = job.server.list_queues()[2]
    elif len(job.structure) > 10 and len(job.structure) <= 15:
        job.server.queue = job.server.list_queues()[3]
    elif len(job.structure) > 15 and len(job.structure) <= 20:
        job.server.queue = job.server.list_queues()[4]
    elif len(job.structure) > 20 and len(job.structure) <= 25:
        job.server.queue = job.server.list_queues()[5]
    elif len(job.structure) > 25 and len(job.structure) <= 30:
        job.server.queue = job.server.list_queues()[6]
    elif len(job.structure) > 30:
        job.server.queue = job.server.list_queues()[7]


    if 'k_mesh' in dic.keys():
        if dic['k_mesh'] is list:
            job.k_mesh = dic['k_mesh']
        else:
            job.k_mesh_spacing = dic['k_mesh']
    else:
        work_str = Kpoints().automatic_density(ase_to_pmg(
            job.structure), kppa=KPPA)
        
        print('KPOINTS file')
        print(work_str)
        
        with open('KPOINTS', 'w') as file:
            file.write(str(work_str))
        job.copy_file_to_working_directory('KPOINTS')

        # job.set_kpoints(mesh=work_str.as_dict()[
        #    'kpoints'][0], center_shift=[0, 0, 0])
        # job.set_kpoints_file(method='Gamma', 
        #    size_of_mesh=mesh=work_str.as_dict()['kpoints'][0])
    job.run()
    if os.path.exists('KPOINTS'):
        os.remove('KPOINTS')
