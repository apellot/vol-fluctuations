# vol-fluctuations

Things to try next:
* Dropout regularization
* L1/L2 functions
* Early stopping
* Data augmentation
* Use n_coll and see it is a better learning/training parameter
* reduce number of hidden layers?

To run GNN and Transformer data set use:
python scripts/train_cached.py --arch gnn \
    --cache data/processed/cached/urqmd_padded.h5 \
    --inputs data/processed/urqmd_auau_*GeV.h5 \
    --output-tag urqmd_v1 \
    --truth-dir data/processed/urqmd_truth/ \
    --num-workers 4 \
    --num-threads 6 \
    --batch-size 32

    Adjust number of threads to your machine


Testing/validation to try:
* error vs. iteration
* loss vs. iteration
* cross entropy loss over epoch