要把项目放在 AutoDL 的 `autodl-tmp` 目录下：否则 `model.save_pretrained_merged()` 保存文件时，会在项目目录下保存临时文件，导致系统盘被占满，无法成功保存。
