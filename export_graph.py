import matplotlib.pyplot as plt

def export_graph(path, epoch, train_loss, train_acc, test_loss, test_acc):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    axL.set_title('model Loss')
    axL.set_xlabel('epochs')
    axL.set_ylabel('loss')
    axR.set_title('model Accuracy')
    axR.set_xlabel('epochs')
    axR.set_ylabel('accuracy')
    axR.grid(True)

    axL.plot(train_loss, label='train', linestyle='solid', lw=0.8)
    axL.plot(test_loss , label='test' , linestyle='solid', lw=0.8)
    axR.plot(train_acc , label='train', linestyle='solid', lw=0.8)
    axR.plot(test_acc  , label='test' , linestyle='solid', lw=0.8)
    #グリッドの設定
    axL.xaxis.grid(linestyle='-', lw=0.5, alpha=0.4, color='lightgray')
    axL.yaxis.grid(linestyle='-', lw=0.5, alpha=0.4, color='lightgray')
    axR.xaxis.grid(linestyle='-', lw=0.5, alpha=0.4, color='lightgray')
    axR.yaxis.grid(linestyle='-', lw=0.5, alpha=0.4, color='lightgray')
    #枠・メモリ設定
    axL.tick_params(direction='in')
    axR.tick_params(direction='in')
    axL.set_xlim([0,epoch])
    axL.set_ylim([0,max(train_loss+test_loss)*1.1])
    axR.set_xlim([0,epoch])
    axR.set_ylim([0,max(train_acc+test_acc)*1.1])
    #凡例を追加
    axL.legend(loc = 'upper right')
    axR.legend(loc = 'upper left')
    #ファイルセーブ
    fig.savefig(path + 'LossAccuracy.png')
    plt.close()
