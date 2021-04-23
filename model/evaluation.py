from model.train import *
import matplotlib.pyplot as plt


def roc_and_prc_to_image(tp_list, fn_list, fp_list, tn_list, illustration):
    fpr_list = []
    tpr_list = []
    for i in range(len(tp_list)):
        tpr_list.append(float(tp_list[i]) / (float(tp_list[i] + fn_list[i])))
        fpr_list.append(float(fp_list[i]) / (float(fp_list[i] + tn_list[i])))

    plt.plot(fpr_list, tpr_list)
    plt.title('ROC')
    plt.xlabel('f p r')
    plt.ylabel('t p r')
    roc_path = model_save_path + "roc" + illustration + get_cur_time() + '.png'
    plt.savefig(roc_path)
    plt.show()

    recall_list = []
    precision_list = []
    for i in range(len(tp_list)):
        recall_list.append(float(tp_list[i]) / (float(tp_list[i] + fn_list[i])))
        precision_list.append(float(tp_list[i]) / (float(tp_list[i] + fp_list[i])))

    plt.plot(recall_list, precision_list)
    plt.title('PRC')
    plt.xlabel('precision')
    plt.ylabel('recall')
    prc_path = model_save_path + "prc" + illustration + get_cur_time() + '.png'
    plt.savefig(prc_path)
    plt.show()


def compute_accuracy_roc(predictions, labels):
    """
    计算roc精确度
    真-真对的标签是1
    真-假对的标签是0
    """

    dmax = np.max(predictions)
    dmin = np.min(predictions)

    step = 0.01
    max_acc = 0
    best_thresh = -1

    tp_list = []
    fn_list = []
    fp_list = []
    tn_list = []

    for d in np.arange(dmin, dmax + step, step):
        # 被标记为真的下标序列
        # 被标记为假的下标序列

        pre_list = predictions.ravel() <= d

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i, pre in enumerate(pre_list):
            if pre == True and labels[i] == 1:
                tp = tp + 1
            elif pre == True and labels[i] == 0:
                fp = fp + 1
            elif pre == False and labels[i] == 0:
                tn = tn + 1
            elif pre == False and labels[i] == 1:
                fn = fn + 1
            else:
                print("出错！")

        # 被标记为真的并且为真的个数
        tpr = float(tp) / (tp + fn)
        # 被标记为假的并且为假的个数
        tnr = float(tn) / (fp + tn)
        acc = 0.5 * (tpr + tnr)
        if acc > max_acc:
            max_acc, best_thresh = acc, d

        # tp 真-真 被判做真的
        # fp 真-假 被判做真的
        # tn 真-假 被判做假的
        # fn 真-真 被判做假的
        tp_list.append(tp)
        fn_list.append(fn)
        fp_list.append(fp)
        tn_list.append(tn)
    return max_acc, best_thresh, tp_list, fn_list, fp_list, tn_list


def compute_accuracy(predictions, labels, threshold):
    d = threshold
    pre_list = predictions.ravel() <= d

    # tp 真-真 被判做真的
    # fp 真-真 被判做假的
    # tn 真-假 被判做假的
    # fn 真-假 被判做真的

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, pre in enumerate(pre_list):
        if pre == True and labels[i] == 1:
            tp = tp + 1
        elif pre == True and labels[i] == 0:
            fp = fp + 1
        elif pre == False and labels[i] == 0:
            tn = tn + 1
        elif pre == False and labels[i] == 1:
            fn = fn + 1
        else:
            print("出错！")

    # 被标记为真的并且为真的个数
    tpr = float(tp) / (tp + fn)
    # 被标记为假的并且为假的个数
    tnr = float(tn) / (fp + tn)
    # 准确率
    acc = 0.5 * (tpr + tnr)
    return acc, tp, fp, tn, fn


def display_result(g_g_pair, g_f_pair, threshold=-1, samples=1000, illustration=""):
    test_gen = generate_batch(g_g_pair, g_f_pair, 1)
    pred, tr_y = [], []

    for i in range(samples):
        (img1, img2), label = next(test_gen)
        tr_y.append(label)
        pred.append(model.predict([img1, img2])[0][0])

    tr_acc, tra_threshold, tp_list, fn_list, fp_list, tn_list = compute_accuracy_roc(np.array(pred), np.array(tr_y))
    if threshold < 0:
        final_threshold = tra_threshold
    else:
        final_threshold = threshold
    print(illustration)
    roc_and_prc_to_image(tp_list, fn_list, fp_list, tn_list, illustration)
    print("最佳准确率：", tr_acc, "最佳阈值：", tra_threshold)
    acc, tp, fp, tn, fn = compute_accuracy(np.array(pred), np.array(tr_y), final_threshold)
    print("准确率为：", acc)
    print("真-真 被判做真的个数有：", tp)
    print("真-真 被判做假的个数有：", fn)
    print("真-假 被判做假的个数有：", tn)
    print("真-假 被判做真的个数有：", fp)


if __name__ == "__main__":
    model.load_weights(model_save_path + '/osv-025.h5')
    display_result(tra_g_g_pair, tra_g_f_pair, -1, 1000, "-随机伪造-训练集-")
    display_result(val_g_g_pair, val_g_f_pair, final_threshold, 1000, "-随机伪造-验证集-")
    display_result(tes_g_g_pair, tes_g_f_pair, final_threshold, 1000, "-随机伪造-测试集-")
    display_result(tra_g_g_pair, tra_g_h_pair, final_threshold, 1000, "-刻意伪造-训练集-")
    display_result(val_g_g_pair, val_g_h_pair, final_threshold, 1000, "-刻意伪造-验证集-")
    display_result(tes_g_g_pair, tes_g_h_pair, final_threshold, 1000, "-刻意伪造-测试集-")
