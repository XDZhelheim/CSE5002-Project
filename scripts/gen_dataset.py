import numpy as np

# x: (N, channels)
# y: (N, 1)


def gen_adj(file_path="../raw_data/adjlist.csv"):
    with open(file_path, "r") as f:
        lines = f.readlines()

    N = len(lines)
    adj = np.zeros((N, N))

    for i in range(N):
        line = lines[i].strip().split(",")
        for node in line:
            if node:  # 不是空字符串
                adj[i, int(node)] = 1  # 这里自己到自己也连了

    print(adj.shape)
    print("All degrees:", (adj == 1).sum())
    print(adj)
    np.savez_compressed("../data/adj.npz", data=adj)

    return adj


def gen_data(file_path="../raw_data/attr.csv"):
    data = np.loadtxt(file_path, delimiter=",", usecols=[1, 2, 3, 4, 5, 6])

    print(data.shape)
    print(data)
    np.savez_compressed("../data/data.npz", data=data)

    return data


def gen_labels(
    train_label_path="../raw_data/label_train.csv",
    test_label_path="../raw_data/label_test.csv",
    train_num=3500,
    val_num=500,
):
    train_labels = np.loadtxt(train_label_path, delimiter=",").astype(np.int16)
    test_labels = np.loadtxt(test_label_path, delimiter=",").astype(np.int16)

    labels = np.concatenate([train_labels, test_labels])
    labels = labels[:, 1]
    labels -= 2000
    labels[labels < 0] = -1

    print("num_class:", len(np.unique(labels))-1)
    print(np.unique(labels))

    # 这里将小于 2000 的标签全部标为 -1
    # 就是直接不要了，这些是一定预测错的
    # 在训练的时候 -1 的标签不参与loss (mask 掉)，但是注意 val 和 test 不能 mask
    # 否则就是算准确率的时候故意让异常值不参与计算，属于严重错误

    print(labels.shape)
    print(labels)
    np.savez_compressed("../data/label.npz", data=labels)

    train_labels = labels[:train_num]

    train_indices = np.argwhere(train_labels != -1).squeeze()
    val_indices = np.arange(train_num, train_num + val_num)
    test_indices = np.arange(train_num + val_num, len(labels))

    print(train_indices)
    print(val_indices)
    print(test_indices)

    np.savez_compressed(
        "../data/indices.npz", train=train_indices, val=val_indices, test=test_indices
    )

    return labels


if __name__ == "__main__":
    gen_adj()
    gen_data()
    gen_labels()
