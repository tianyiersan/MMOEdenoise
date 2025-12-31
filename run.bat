@echo off

rem Set the values for hyperparameters
rem set lr_list="0.01 0.005 0.001 0.0005"
rem set reg_weight_list="0.01 0.001 0.0001"
set emb_size=64
set lr=0.01
set reg_weight=1e-3

rem Define the dataset names
set dataset_list="tmall beibei"

rem Iterate over dataset names
for %%D in (%dataset_list%) do (
    rem Iterate over lr values
    rem for %%L in (%lr_list%) do (
        rem Iterate over reg_weight values
        rem for %%R in (%reg_weight_list%) do (
            rem Iterate over emb_size values
            for %%E in (%emb_size%) do (
                echo "start train: %%D"
                python main.py ^
                    --lr %lr% ^
                    --reg_weight %reg_weight% ^
                    --data_name %%D ^
                    --embedding_size %%E
                echo "train end: %%D"
            )
        )
    rem )
rem )
