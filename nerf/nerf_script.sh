#!/bin/bash -e

pyenv global tensorflow2

echo Expirement Name $1
echo Basedir $2
echo Datadir $3

python run_nerf.py --expname $1 --basedir $2 --datadir $3 --use_viewdirs --N_importance 128  --no_ndc --lindisp
