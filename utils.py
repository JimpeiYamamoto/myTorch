import os
import glob
import shutil
import random
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

    def test(net, test_loader, heatmap_path, device, criterion):
        '''
        net: 学習済みモデル
        test_loader: test_loader
        heatmap_path: 作成したヒートマップの保存パス
        device: device
        criterion: 最適化関数
        '''
        test_loss = 0.0
        test_acc = 0.0
        net.eval()
        with torch.no_grad():
            for _, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss =  loss.item() / len(test_loader.dataset)
            test_acc = (outputs.max(1)[1] == labels).sum() / len(test_loader.dataset)
        print("test_loss: {0:4f}, test_acc: {1:4f}".format(test_loss, test_acc))
        cm = confusion_matrix(labels, outputs.argmax(1))
        sns.heatmap(cm)
        plt.savefig(heatmap_path)
        return outputs

    def split_correct_image_csv(test_df, outcome_df_path, correct_path, incorrect_path, custom_test_dataset, outputs, labels):
        '''
        correct_path: 正解だった画像の保存先
        incorrect_path: 不正解だった画像の保存先
        custom_test_dataset: テストカスタムデータセット
        outputs: テストの結果
        labels: テストデータのラベル
        '''
        correct_lst = []
        for i, image_path in enumerate(custom_test_dataset.images):
            output = outputs[i].argmax()
            file_name = os.path.basename(image_path)
            if output == labels[i]:
                dst = correct_path
                correct_lst.append(1)
            else:
                dst = incorrect_path
                correct_lst.append(0)
            shutil.copyfile(image_path, os.path.join(dst, file_name))
        df = test_df.copy()
        df['label'] = labels
        df['outputs'] = outputs.argmax(1)
        df['iscorrect'] = correct_lst
        df.to_csv(outcome_df_path)

    def plot_epoch_to_loss_acc(num_epoch, train_loss_list, val_loss_list, train_acc_list, val_acc_list):
        plt.figure()
        plt.plot(range(num_epoch), train_loss_list, color='blue', linestyle='-', label = 'train_loss')
        plt.plot(range(num_epoch), val_loss_list, color='green', linestyle='--', label = 'val_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and validation loss')
        plt.grid()
    
        plt.figure()
        plt.plot(range(num_epoch), train_acc_list, color='blue', linestyle='-', label='train_acc')
        plt.plot(range(num_epoch), val_acc_list, color='green', linestyle='--', label='val_acc')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Training and validation acc')
        plt.grid()