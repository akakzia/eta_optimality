import os
from time import sleep

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = os.environ['SCRATCH']

# Make top level directories
mkdir_p(job_directory)

nb_seeds = 5
agents = ['SAC', 'GSAC']
envs = ['HalfCheetah-v2', 'Ant-v2']

for i in range(nb_seeds):
    for agent in agents:
        for env in envs:
            job_file = os.path.join(job_directory, "{}_{}_{}.slurm".format(agent, env, i))

            with open(job_file, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --account=oke@cpu\n")
                fh.writelines("#SBATCH --job-name={}_{}\n".format(agent, env))
                fh.writelines("#SBATCH --partition=cpu_p1\n")
                fh.writelines("#SBATCH --qos=qos_cpu-t4\n")
                fh.writelines("#SBATCH --output={}_{}_%j.out\n".format(agent, env))
                fh.writelines("#SBATCH --error={}_{}_%j.out\n".format(agent, env))
                fh.writelines("#SBATCH --time=40:00:00\n")
                fh.writelines("#SBATCH --nodes=1\n")
                fh.writelines("#SBATCH --ntasks=2\n")
                fh.writelines("#SBATCH --hint=nomultithread\n")

                fh.writelines("module load pytorch-cpu/py3/1.4.0\n")

                fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
                fh.writelines("export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
                fh.writelines("export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/include\n")
                fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uqy56ga/.mujoco/mujoco200/bin\n")

                fh.writelines("srun python -u -B main.py --algo {} --env-name {} --save-dir 'experiments_{}' 2>&1".format(agent, env, env))

            os.system("sbatch %s" % job_file)
            sleep(1)