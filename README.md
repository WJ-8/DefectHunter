# DefectHunter
## note: cuda 11.7 cudnn 8.5

## dataset process

```shell
python process/build_ast.py
python process/build_cfgdfg.py
python process/bulid_y.py
python process/Specialword.py
python process/build_pls.py
```

## train

```shell
python train.py
```
