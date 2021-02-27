# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh # sample code to activate conda in bash
conda create -n tattletale python=3.7 -y
conda activate tattletale

pip install transformers torch