# SE(3) diffusion model with application to protein backbone generation

[![standard-readme compliant](https://img.shields.io/badge/SE3%20StrcturePrediction%20-init-green.svg?style=plastic&logo=appveyor)](https://github.com/Wangchentong/se3_diffusion)
[![standard-readme compliant](https://img.shields.io/badge/SE3%20ComplexDiffusion%20-init-green.svg?style=plastic&logo=appveyor)](https://github.com/Wangchentong/se3_diffusion)
[![standard-readme compliant](https://img.shields.io/badge/SE3%20MoleculeDiffusion-Proposed-inactive.svg?style=plastic&logo=appveyor)](https://github.com/Wangchentong/se3_diffusion)

Implementation for "SE(3) diffusion model with application to protein backbone generation" [arxiv link](https://arxiv.org/abs/2302.02277).

Code based on official github repo [SE3 Diffusion](https://github.com/jasonkyuyim/se3_diffusion/)


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Dataset Preparation](#dataset-preparation)
	- [Standard Diffusion Data](#standard-diffusion-data)
	- [Structure Prediction Data](#structure-prediction-data)
- [Training](#training)
  - [Standard Diffusion Training](#standard-diffusion-training)
  - [Structure Prediction Diffusion Training](#structure-prediction-diffusion-training)
- [Inference](#inference)
  - [Standard Diffusion Inference](#standard-diffusion-inference)
<!--   - [Structure Prediction Inference](#structure-prediction-inference) -->
- [Update History](#update-history)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background


## Install

We recommend [miniconda](https://docs.conda.io/en/main/miniconda.html) (or anaconda).
Run the following to install a conda environment with the necessary dependencies.
```bash
conda env create -f se3.yml
```

Next, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```

[Pyrosetta](https://graylab.jhu.edu/download/PyRosetta4/archive/release/) is recommended to install for best performance of mpnn sequence design, {Username} {Password} of License can be obtained [here](https://els2.comotion.uw.edu/product/pyrosetta)
```
aria2c https://{Username}:{Password}@graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python39.linux/PyRosetta4.Release.python39.linux.release-342.tar.bz2

tar -xvjf PyRosetta4.Release.python39.linux.release-342.tar.bz2

conda activate se3

cd PyRosetta4.Release.python39.linux.release-342

python setup/setup.py install
```

## Dataset Preparation


### Standard Diffusion Data

```sh
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ ./data/mmCIF

gzip -d ./data/mmCIF/**/*.gz

# --dump_chain_pdb --dump_chain_fasta (dump pdb and fatsa of single chain for further clustring)
# --mode update (update metadata and processed pickel without overwritten old)
python data/process_pdb_dataset.py --num_processes 20
```

See the script for more options. Each mmCIF will be written as a pickle file that we read and process in the data loading pipeline. A metadata.csv will be saved that contains unique assemble unit defined by [pdbx_struct_assembly](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/pdbx_struct_assembly.html) in each mmcif file.

### Structure Prediction Data

Download templates of each PDB single chain
```sh
# install aws
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

./aws/install --bin-dir ~/bin --install-dir ~/local/aws-cli --update

# Download OpenProteinSet pdb templates data
mkdir strcture_prediction/OpenProteinSet/

aws s3 sync --no-sign-request s3://openfold/pdb/ ./strcture_prediction/OpenProteinSet/ --exclude "**/**.a3m"
```
Process OpenProteinSet templates file with data/mmCIF cif file.   
```sh
# fix discripency of pdb name between mmCIF and OpenProteinSet
wget https://ftp.pdbj.org/pub/pdb/data/status/obsolete.dat  ./strcture_prediction/data/mmCIF/
# data preprocess
python structure_prediction/data/process_pdb_dataset.py --num_processes 20 
```
it shoulbe noted that **Standard Diffusion Data** suggested to be prepared brefore process strcture prediction data. As it save template release date cache in data/metadata.csv. This cache can boost at least 2 times speed up.

## Training

### Standard Diffusion Training
Train scaffold generation model with 4 gpu in DDP mode, log with W&B
```sh
torchrun --nproc_per_node 4 experiments/train_se3_diffusion.py experiment.num_gpus=4 experiment.use_ddp=True experiment.use_wandb=True
```
If you want to train on oligomer data rather than only monomer(it will make dataset 2x size than only monomer) , append this parameter
```sh
# more oligomer type can be appended, view full ologomer type of pdb in the config/base.yaml(keyword:allowed_oligomer)
data.filtering.allowed_oligomer=["monomer","dimer"]
```
<!-- ### Structure Prediction Training -->

### Structure Prediction Diffusion Training
Train scaffold generation model with 4 gpu in DDP mode, log with W&B
```sh
torchrun --nproc_per_node 4 experiments/train_se3_diffusion.py experiment.num_gpus=4 experiment.use_ddp=True experiment.use_wandb=True
```
<!-- ### Structure Prediction Training -->

## Inference

### Standard Diffusion Inference

```sh
python experiments/inference_se3_diffusion.py
```
if Pyrosetta is installed, following config can be used to improve scaffold generation quality
```yaml
# ./data/inference.yaml
inference:
  mpnn:
    # if use ca_mpnn weights
    ca_only : False
    # Whether optimize backbone by rosetta fades
    fades : False
    # Relax cycle for each scaffold, min cycle is 1 for sidechain pack, 0 will only give sequence
    cycle : 1
    # Select best quence of multi sample sequences on current scaffold based on mpnn score rather than direct sample one sequence
    best_selection:
      switch_on : False
      sample_nums : 32
    # whether dump full-atom protein after mpnn sequence design
    dump: True
    # Look more config in this file
```
Inference output would be like
```shell
inference_outputs
└── 12D_02M_2023Y_20h_46m_13s           # Date time of inference.
    ├── inference_conf.yaml             # Config used during inference.
    ├── mpnn.fasta                      # mpnn designed seuences.
    ├── self_consistency.csv            # self consistency analysis, contains rmsd and tmscore between scaffold ans esmfold, mpnn score of sequence, scaffold path, esmf path etc.
    ├── diffusion                       # dir contains scaffold generated by framediff
    │    ├── 100_1_sample.pdb          
    │    ├── 100_2_sample.pdb           # {length}_{sample_id}_sample.pdb
    |    └── ...
    ├── trajctory                       # dir contains traj pdb, exists when inference.option.save_trajactory=True
    │    ├── 100_1_bb_traj.pdb          
    │    ├── 100_2_bb_traj.pdb          # {length}_{sample_id}_traj.pdb
    |    └── ...
    ├── movie                           # dir contains full atom protein designed by mpnn, exists when inference.option.plot.switch_on=True
    │    ├── 100_1_rigid_movie.gif      # movie of protein rigid at time t    
    │    ├── 100_1_rigid_0_movie.gif    # movie of predict protein rigid at time 0 from time t  
    |    └── ...
    ├── mpnn                            # dir exists when pyrosetta in installed and inference.mpnn.dump=True
    │    ├── 100_0_sample_mpnn_0.pdb      
    │    ├── 100_0_sample_mpnn_1.pdb    # {length}_{sample_id}_sample_mpnn_{sequence_id}.pdb
    |    └── ... 
    └── esmf                            # dir contians esmf predict strcture
         ├── 100_0_sample_esmf_0.pdb     
         ├── 100_0_sample_esmf_0.pdb     # {length}_{sample_id}_sample_esmf_{sequence_id}.pdb
         └── ... 

```

A naive benchmark of a couple of parameter combination(more combination strategy will be updated)
```sh
# higher inference.cpu_num can boost speed of pyrosetta scaffold optimization by multiprocessing
baseline : python experiments/inference_se3_diffusion.py

baseline_ca : python experiments/inference_se3_diffusion.py inference.name=baseline_ca_only inference.mpnn.ca_only=True

cycle_3 : python experiments/inference_se3_diffusion.py inference.name=mpnn_cycle_3 inference.mpnn.cycle=3 inference.cpu_num=14

cycle_3_best_selection : python experiments/inference_se3_diffusion.py inference.name=mpnn_cycle_3_best_selection inference.mpnn.cycle=3 inference.mpnn.best_selection.switch_on=True inference.cpu_num=14

cycle_3_fades : python experiments/inference_se3_diffusion.py inference.name=mpnn_cycle_3_fades inference.mpnn.cycle=3 inference.mpnn.fades=True inference.cpu_num=14
```

<div align="center">
  <img src="https://github.com/Wangchentong/se3_diffusion/assets/91596060/a1056108-c656-40b1-af8c-b3ea957e4de3" alt="mpnn_pyrosetta">
</div>

<!-- ### Structure Prediction Inference -->

## Update History
### Structure Prediction
 * 2023.6.14 Fisrt release version of strcture prediction pretraining with diffusion model
### Standard Diffusion
 * 2023.6.14 :  
 	1. Add The Standard diffusion data process and training can comprimise oligomer.
	2. Change set deafult self-condition embedder to template embedder(include frame rotation and torsion angle(not used now), use triagular update, show promising performence improvement)   
	3. Fix Bug self-condition is trained on the output of current step in previous version

## Related Efforts

- [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion) - 💌 The Base Code of this repo.
- [Openfold](https://github.com/aqlaboratory/openfold) - 💵 The Base model and OpenProteinSet dataset Provider.
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - ♥ The sequence design model.
- [DL_binder_design](https://github.com/nrbennet/dl_binder_design) 📨 The ProteinMPNN-Pyrosetta optimization method.
- [ESMFold](https://github.com/facebookresearch/esm) 🔦 The structure prediction model.

## Maintainers

[@Wangchentong](https://github.com/Wangchenton).

## Contributing

Feel free to join in this repo! [Open an issue](https://github.com/Wangchentong/se3_diffusion/issues/new) or submit PRs.

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## License

[MIT](LICENSE) © Wang Chentong
