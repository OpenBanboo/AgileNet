# Fewshot Dictionary Learning on MNIST

```bash
pip install -r requirements.txt
python main_fewshot.py --ori_train=True --dic_train=True --fewshot_train=True --save_ori=True --save_decomp=True --epochs=10 --epochs_decomp=1  --epochs_fewshot 10
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --test-batch-size N   input batch size for testing (default: 1000)
  --lr LR               learning rate (default: 0.01)
  --momentum M          SGD momentum (default: 0.5)
  --no-cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --log-interval N      how many batches to wait before logging training
                        status
  --save IS_SAVE        save the model and weights
  --get_rid_of GET_RID_OF
                        The number to get rid of.
  --ori_train IS_ORI_TRAIN
                        Execute original training with OCO_Net (Please input
                        True/False, default: False)
  --dic_train IS_DIC_TRAIN
                        Use diction training with DD_Net (Please input
                        True/False, default: False)
  --fewshot_train IS_FEWSHOT_TRAIN
                        Execute fewshot training with DDF_Net (Please input
                        True/False, default: False)
  --save_ori IS_SAVE_ORI
                        Save the trained original trained model? (Please input
                        True/False, default: False)
  --save_decomp IS_SAVE_DECOMP
                        Save the trained decomposed trained model? (Please
                        input True/False, default: False)
  --epochs N            number of epochs to train original net (default: 10)
  --epochs_decomp N     number of epochs to train dictionary decomposed net
                        (default: 1)
  --epochs_fewshot N    number of epochs to fewshot train dictionary
                        decomposed net (default: 1)
```
