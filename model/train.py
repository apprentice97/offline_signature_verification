from model.model import *


def get_cur_time():
    curr_time = datetime.datetime.now()
    return str(curr_time.strftime("%m%d%H%M"))


# 绘制损失函数曲线
def loss_curve(his=""):
    plt.semilogy(his.history['loss'], color='b', label="Training loss")
    plt.semilogy(his.history['val_loss'], color='r', label="validation loss")
    plt.title("chinese dataset")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    roc_path = model_save_path + get_cur_time() + '.png'
    plt.legend()
    plt.savefig(roc_path)
    plt.show()


if __name__ == "__main__":
    print("-----------打印训练 history-------")
    history = model.fit(generate_batch(tra_g_g_pair + tra_g_g_pair, tra_g_f_pair + tra_g_h_pair, batch_sz),
                        steps_per_epoch=100,
                        validation_steps=100,
                        epochs=epochs,
                        validation_data=generate_batch(val_g_g_pair + val_g_g_pair, val_g_f_pair + val_g_h_pair,
                                                       batch_sz),
                        callbacks=callbacks)
    print(history)
    loss_curve(history)
