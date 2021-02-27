source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda create -n that_one python=3.7 -y
conda activate that_one

pip install transformers torch