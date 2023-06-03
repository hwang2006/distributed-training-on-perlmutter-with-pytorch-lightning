# Distributed-training-on-perlmutter-with-pytorch-lightning

This repository is intended to guide users to run their pytorch lightning codes on Neuron. [Pytorch Lightning](https://www.pytorchlightning.ai/index.html) is a lightweight wrapper or interface on top of PyTorch, which simplifies the implementation of complex deep learning models. It is a PyTorch extension that enables researchers and practitioners to focus more on their research and less on the engineering aspects of deep learning. PyTorch Lightning automates many of the routine tasks, such as distributing training across multiple GPUs, logging, and checkpointing, so that the users can focus on writing high-level code for their models.

**Contents**
* [NERSC Perlmutter Supercomputer](#nersc-perlmutter-supercomputer)
* [Installing Conda](#installing-conda)
* [Installing Pytorch Lightning](#installing-pytorch-lightning)
* [Running Jupyter](#running-jupyter)
* [Pytorch Lightning Examples on Jupyter](#pytorch-lightning-examples-on-jupyter) 
* [Running Pytorch Lightning on SLURM](#running-pytorch-lightning-on-slurm)

## NERSC Perlmutter Supercomputer
[Perlmutter](https://docs.nersc.gov/systems/perlmutter/), located at [NERSC](https://www.nersc.gov/) in [Lawrence Berkeley National Laboratory](https://www.lbl.gov/), is a HPE Cray EX supercomputer with ~1,500 AMD Milan CPU nodes and ~6000 Nvidia A100 GPUs (4 GPUs per node). It debuted as the world 5th fastest supercomputer in the Top500 list in June 2021. Please refer to [Perlmutter Architecture](https://docs.nersc.gov/systems/perlmutter/architecture/) for the architecutural details of Perlmutter including system specifications, system performance, node specifications and interconnect. [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling. 

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218645916-30e920b5-b2cf-43ad-9f13-f6a2568c0e37.jpg" width=550/></p>

## Installing Conda
Once logging in to Perlmutter, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

Note that we will assume that your NERSC account information is as follows:
```
- Username: elvis
- Project Name : ddlproj
- Account (CPU) : m1234
- Account (GPU) : m1234_g
```
so, you will have to replace it with your real NERSC account information.

1. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
perlmutter:login15>$ cd $SCRATCH 
perlmutter:login15>$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```
```
# (option 2) Miniconda 
perlmutter:login15>$ cd $SCRATCH
perlmutter:login15>$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
2. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. Please refer to [File System Overview](https://docs.nersc.gov/filesystems/) for more details of NERSC storage systems . You will install it on the /global/common/software/&lt;myproject&gt; directory and then create a conda virtual environment on your own scratch directory. 
```
perlmutter:login15>$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
perlmutter:login15>$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/global/homes/s/swhwang/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/global/homes/s/swhwang/miniconda3] >>> /global/common/software/ddlproj/elvis/miniconda3  <======== type /global/common/software/myproject/$USER/miniconda3 here
PREFIX=/global/common/software/dasrepo/swhwang/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /global/common/software/ddlproj/elvis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
If you'd prefer that conda's base environment not be activated on startup,
   set the auto_activate_base parameter to false:

conda config --set auto_activate_base false

Thank you for installing Miniconda3!
```

3. finalize installing Miniconda with environment variables set including conda path.
```
perlmutter:login15>$ source ~/.bashrc    # set conda path and environment variables 
perlmutter:login15>$ conda config --set auto_activate_base false
perlmutter:login15>$ which conda
/global/common/software/ddlproj/elvis/miniconda3/condabin/conda
perlmutter:login15>$ conda --version
conda 23.1.0
```

## Installing Pytorch Lightning
Now you are ready to build Horovod as a conda virtual environment: 
1. load modules: 
```
perlmutter:login15>$ module load cudnn/8.3.2 nccl/2.17.1-ofi evp-patch
```
2. create a new conda virtual environment and activate the environment:
```
perlmutter:login15>$ conda create -n lightning
perlmutter:login15>$ conda activate lightning
```
3. install the pytorch and lightning package:
```
(lightning) perlmutter:login15>$ conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
(lightning) perlmutter:login15>$ pip install lightning
```
4. check if the pytorch lightning packages were installed:
```
(lightning) [glogin01]$ conda list | grep lightning
# packages in environment at /scratch/$USER/miniconda3/envs/lightning:
lightning                 2.0.2                    pypi_0    pypi
lightning-cloud           0.5.36                   pypi_0    pypi
lightning-utilities       0.8.0                    pypi_0    pypi
pytorch-lightning         2.0.2                    pypi_0    pypi
```

## Running Jupyter
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. Jupyterlab is the latest web-based interactive development environment for notebooks, code, and data. The Jupyter Notebook is the original web application for creating and sharing computational documents. NERSC provides a [JupyterHub service](https://docs.nersc.gov/services/jupyter/#jupyterhub) that allows you to run a Jupyter notebook on Perlmutter. JupyterHub is designed to serve Jupyter notebook for multiple users to work in their own individual workspaces on shared resources. Please refer to [JupyterHub](https://jupyterhub.readthedocs.io/en/stable/) for more detailed information. You can also run your own jupyter notebook server on a compute node (*not* on a login node), which will be accessed from the browser on your PC or laptop through SSH tunneling. 
<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/>
</p>

In order to do so, you need to add the horovod-enabled virtual envrionment that you have created as a python kernel.
1. activate the horovod-enabled virtual environment:
```
perlmutter:login15>$ conda activate horovod
```
2. install Jupyter on the virtual environment:
```
(lightning) perlmutter:login15>$ conda install jupyter
(lightning) perlmutter:login15>$ pip install jupyter-tensorboard
```
3. add the virtual environment as a jupyter kernel:
```
(lightning) perlmutter:login15>$ pip install ipykernel 
(lightning) perlmutter:login15>$ python -m ipykernel install --user --name horovod
```
4. check the list of kernels currently installed:
```
(lightning) perlmutter:login15>$ jupyter kernelspec list
Available kernels:
  pytho       /global/common/software/ddlproj/evlis/miniconda3/envs/craympi-hvd/share/jupyter/kernels/python3
  lightning     /global/u1/s/elvis/.local/share/jupyter/kernels/lightning
```
5. launch a jupyter notebook server on a compute node 
- to deactivate the virtual environment
```
(lightning) perlmutter:login15>$ conda deactivate
```
- to create a batch script for launching a jupyter notebook server: 
```
perlmutter:login15>$ cat jupyter_run.sh
#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -c 32

export SLURM_CPU_BIND="cores"

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the node name and port
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@perlmutter-p1.nersc.gov" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@perlmutter-p1.nersc.gov"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@perlmutter-p1.nersc.gov" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@perlmutter-p1.nersc.gov"

echo "load module-environment"
module load  cudnn/8.7.0  nccl/2.17.1-ofi  evp-patch

echo "execute jupyter"
source ~/.bashrc
conda activate lightning
#cd $SCRATCH/ddl-projects #root/working directory for jupyter server
cd $SCRATCH #root/working directory for jupyter server
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER}
#bash -c "jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token='${USER}'"
echo "end of the job"
```
- to launch a jupyter notebook server 
```
perlmutter:login15>$ sbatch jupyter_run.sh
Submitted batch job 5494200
```
- to check if a jupyter notebook server is running
```
perlmutter:login15>$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5494200  gpu_ss11 jupyter_  swhwan PD       0:00      1 (Priority)
perlmutter:login15>$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            5494200       gpu_ss11  jupyter_    evlis  RUNNING       0:02   8:00:00      1 nid001140
perlmutter:login15>$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://nid######:#####/lab?token=...
.
.
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
perlmutter:login15>$ cat port_forwarding_command
ssh -L localhost:8888:nid######:##### $USER@neuron.ksc.re.kr
```
6. open a SSH client (e.g., Putty, PowerShell, Command Prompt, etc) on your PC or laptop and log in to Perlmutter just by copying and pasting the port_forwarding_command:
```
C:\Users\hwang>ssh -L localhost:8888:nid######:##### elvis@perlmutter-p1.nersc.gov
Password(OTP):
Password:
```
7. open a web browser on your PC or laptop to access the jupyter server
```
- URL Address: localhost:8888
- Password or token: elvis   # your username on Perlmutter
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 

## Pytorch Lightning Examples on Jupyter
Now, you are ready to run a pytorch lightning code on a jupyter notebook. Hopefully, the jupyter notebook examples would lead to getting familiarized yourself to the basics of pytorch lightning coding practices step-by-step. Please refer to the [notebooks](https://github.com/hwang2006/distributed-training-on-perlmutter-with-pytorch-lightning/tree/main/notebooks) directory for example codes.
* [Writing a Pytorch Lightning Simple Example Step-by-Step](https://nbviewer.org/github/hwang2006/distributed-training-on-perlmutter-with-pytorch-lightning/blob/main/notebooks/pytorch_lightning_example.ipynb)
* [Fine-tunning a Pretrained BERT Model for Sentiment Classification](https://nbviewer.org/github/hwang2006/distributed-training-on-perlmutter-with-pytorch-lightning/blob/main/notebooks/pt_bert_nsmc_lightning.ipynb)

## Running Pytorch Lightning on SLURM
We will show how to run a simple pytorch lightning code on multiple nodes interactively.
1. request allocation of available GPU-nodes:
```
(horovod) perlmutter:login15>$ salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node=4 --account=m1234_g
salloc: Pending job allocation 5472214
salloc: job 5472214 queued and waiting for resources                                   
salloc: job 5472214 has been allocated resources                                         
salloc: Granted job allocation 5472214
salloc: Waiting for resource configuration
salloc: Nodes nid[001140-001141] are ready for job
```
2. load modules and activate the horovod conda environment:
```
nid001140>$ module load cudnn/8.3.2 nccl/2.14.3  evp-patch
nid001140>$ conda activate lightning
(lightning) nid001140>$
```
4. run a pytorch lightning code:
- to run on the two nodes with 4 GPUs each
```
(lightning) nid001140>$ srun -N 2 --ntasks-per-node=4 python distributed-training-on-perlmutter-with-pytorch-lightning/src/pytorch_mnist_lightning.py --num_nodes 2
```
- to run the Bert NSMC (Naver Sentiment Movie Corpus) example in the [src](https://github.com/hwang2006/distributed-training-on-perlmutter-with-pytorch-lightning/tree/main/src) directory, you need to install additional packages (i.e., emoji, soynlp, transformers and pandas) and download the nsmc datasets, for example, using git cloning
```
(lightning) nid001140>$ pip install emoji==1.7.0 soynlp transformers pandas
(lightning) nid001140>$ git clone https://github.com/e9t/nsmc  # download the nsmc datasets in the ./nsmc directory
(lightning) nid001140>$ srun -N 2 --ntasks-per-node=4 python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py --num_nodes 2
```
- to run on the two nodes with 2 GPUs each
```
(lightning) nid001140>$ srun -N 2 --ntasks-per-node=2 python distributed-training-on-perlmutter-with-pytorch-lightning/src/pytorch_mnist_lightning.py --num_nodes 2 --devices 2
(lightning) nid001140>$ srun -N 2 --ntasks-per-node=2 python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py --num_nodes 2 --devices 2
```
- to run on the two nodes with 1 GPU each
```
(lightning) nid001140>$ srun -N 2 --ntasks-per-node=1 python distributed-training-on-perlmutter-with-pytorch-lightning/src/pytorch_mnist_lightning.py --num_nodes 2 --devices 1
(lightning) nid001140>$ srun -N 2 --ntasks-per-node=1 python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py --num_nodes 2 --devices 1
```
- to run one node with 4 GPUs
```
(lightning) nid001140>$ python distributed-training-on-perlmutter-with-pytorch-lightning/src/pytorch_mnist_lightning.py 
(lightning) nid001140>$ python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py
(lightning) nid001140>$ srun -N 1 --ntasks-per-node=4 python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py
```
- to run one node with 2 GPUs
```
(lightning) nid001140>$ python distributed-training-on-perlmutter-with-pytorch-lightning/src/pytorch_mnist_lightning.py --devices 2
(lightning) nid001140>$ python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py --devices 2
(lightning) nid001140>$ srun -N 1 --ntasks-per-node=2 python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py --devices 2
```
Now, you are ready to run a pytorch lightning batch job.
1. edit a batch job script running on 4 nodes with 4 GPUs each:
```
perlmutter:login15>$ cat pt_lightning_batch.sh
#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 2
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

export SLURM_CPU_BIND="cores"

module load  cudnn/8.3.2  nccl/2.17.1-ofi  evp-patch
source ~/.bashrc
conda activate craympi-hvd

#srun python pt_bert_nsmc_lightning.py --num_nodes 2
srun python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py --num_nodes 2
```
2. submit and execute the batch job:
```
perlmutter:login15>$ sbatch pt_lightning_batch.sh
Submitted batch job 5473133
```
3. check & monitor the batch job status:
```
perlmutter:login15>$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5473133  gpu_ss11 horovod_    elvis PD       0:00      2 (Priority)
perlmutter:login15>$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5473133  gpu_ss11 horovod_    elvis  R       0:33      2 nid[002836,003928,004010,005024]
```
4. check the standard output & error files of the batch job:
```
perlmutter:login15>$ cat slurm_5473133.out
perlmutter:login15>$ cat slurm_5473133.err
```
5. For some reason, you may want to stop or kill the batch job.
```
perlmutter:login15>$ scancel 5473133
```




