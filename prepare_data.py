import os
import glob
import shutil
import random

class utils():

    def imagepath_to_tu(path, df, img_row_name='img_path', tu_row_name='treated_t'):
        '''
        path: 画像のpath
        df: 水質データのdataframe
        img_row_name: df内の画像の列名
        tu_row_name: df内の処理水濁度の列名
        '''
        file_name = os.path.basename(path)
        col = df[df[img_row_name] == file_name]
        return float(col[tu_row_name])

    def split_class_image(srcs, df, dest0, dest1, dest2):
        '''
        srcs: ソース
        df: 水質データが入ったdataframe
        dest0: < 0.5の画像を保存するpath
        dest1: <= 1.0の画像を保存するpath
        dest2: > 1.0の画像を保存するpath
        '''
        srcs_files = glob.glob(os.path.join(srcs, "*.jpg"))
        for file in srcs_files:
            tu = utils.imagepath_to_tu(file, df)
            if tu < 0.5:
                shutil.move(file, os.path.join(dest0, os.path.basename(file)))
            elif tu <= 1.0:
                shutil.move(file, os.path.join(dest1, os.path.basename(file)))
            else:
                shutil.move(file, os.path.join(dest2, os.path.basename(file)))

    def split_train_validation(dest, srcs, vali_rate):
        '''
        dest: サンプリングしたvalidationファイルの移動先のフォルダ
        srcs: サンプリング前のファイルが保存されているフォルダ
        vali_rate: validationファイルの比率
        '''
        srcs_files = glob.glob(srcs)
        file_len = int(len(srcs_files) * vali_rate)
        for _ in range(file_len):
            srcs_files = glob.glob(srcs)
            file = random.sample(srcs_files, 1)[0]
            shutil.move(file, os.path.join(dest, os.path.basename(file)))
