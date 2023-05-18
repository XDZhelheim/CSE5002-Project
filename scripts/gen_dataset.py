import numpy as np
from tqdm import tqdm

# x: (n*p, num_grid_h, num_grid_w, channels)
# y: (n*p, num_grid_h, num_grid_w) 在路径的点占整个grid里点的比例+在路径的点占整个路径里点的比例

# ! 更新: 不计算占比了, 直接统计数量
# 统计占比 + MAELoss: 不太行, MAE 倾向于预测均值, 而且都是0.0x 太小了
#   所以预测出来每个格子的 y_pred 几乎一样 没有区分性
# 改成统计数量 + MSELoss

# ! 二更: 改成计算百分比 (对于10*10的格子 没区别)
# 还是得适配不同尺寸的格子 做归一化


def gen_grid_xy(images, paths, num_grid_h=20, num_grid_w=20):
    """
    images: (n, h, w)
    paths: (n, num_paths, path_len(not const), 2) !!this is not a matrix

    will generate n*p samples, n=num_images, p=num_paths

    channels:
    - 障碍物占 grid 比例 (float)
    - 是否为起/终所在的格子 (0 or 1)
    - 到起点的距离 (不好用)
    - 到终点的距离 (不好用)
    """
    num_images, image_height, image_width = images.shape
    num_paths = len(paths[0])
    num_channels = 3

    assert image_height % num_grid_h == 0 and image_width % num_grid_w == 0

    grid_height = image_height // num_grid_h
    grid_width = image_width // num_grid_w
    grid_area = grid_height * grid_width

    x = np.zeros((num_images, num_paths, num_grid_h, num_grid_w, 3))
    y = np.zeros((num_images, num_paths, num_grid_h, num_grid_w))
    for i in tqdm(range(num_images)):
        image = images[i]

        # 统计每个格子障碍物点数占比
        for idxx in range(image.shape[0]):
            for idxy in range(image.shape[1]):
                if image[idxx, idxy] == 1:
                    x[i, :, idxx // grid_height, idxy // grid_width, 0] += 1
        # x[i, :, :, :, 0] /= grid_area
        x[i, :, :, :, 0] = x[i, :, :, :, 0] / grid_area * 100

        path_list = paths[i]
        for j in range(len(path_list)):
            path = path_list[j]  # (path_len, 2)
            start_point = path[0]
            end_point = path[-1]

            # 标记起点格 终点格
            x[i, j, start_point[0] // grid_height, start_point[1] // grid_width, 1] = 1
            x[i, j, end_point[0] // grid_height, end_point[1] // grid_width, 2] = 1

            # # 起点终点距离
            # for h in range(num_grid_h):
            #     for w in range(num_grid_w):
            #         x[i, j, h, w, 1] = ((h - start_point[0])**2 + (w - start_point[1])**2)**0.5
            #         x[i, j, h, w, 2] = ((h - end_point[0])**2 + (w - end_point[1])**2)**0.5

            for point in path:
                y[i, j, point[0] // grid_height, point[1] // grid_width] += 1
            # 计算 y
            # y[i, j] = y[i, j] / grid_area + y[i, j] / len(path)
            y[i, j] = y[i, j] / grid_area * 100

    print("x:", x.shape)
    print("y:", y.shape)

    x = x.reshape(num_images * num_paths, num_grid_h, num_grid_w, num_channels)
    y = y.reshape(num_images * num_paths, num_grid_h, num_grid_w)

    return x, y


def gen_grid_dataset(n, p, num_grid_h=20, num_grid_w=20, train_size=0.7, val_size=0.1):
    images = np.load(f"../data/n_{n}_p_{p}/image_{n}_{p}.npz")["data"]
    paths = np.load(f"../data/n_{n}_p_{p}/path_{n}_{p}.npz", allow_pickle=True)["data"]

    x, y = gen_grid_xy(images, paths, num_grid_h, num_grid_w)

    num_samples = x.shape[0]
    split1 = int(num_samples * train_size)
    split2 = int(num_samples * (train_size + val_size))

    x_train = x[:split1]
    x_val = x[split1:split2]
    x_test = x[split2:]

    y_train = y[:split1]
    y_val = y[split1:split2]
    y_test = y[split2:]

    print("x_train:", x_train.shape)
    print("x_val:", x_val.shape)
    print("x_test:", x_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    np.savez_compressed(
        f"../data/n_{n}_p_{p}/train_{n}_{p}.npz", x_train=x_train, y_train=y_train
    )
    np.savez_compressed(
        f"../data/n_{n}_p_{p}/val_{n}_{p}.npz", x_val=x_val, y_val=y_val
    )
    np.savez_compressed(
        f"../data/n_{n}_p_{p}/test_{n}_{p}.npz", x_test=x_test, y_test=y_test
    )


if __name__ == "__main__":
    gen_grid_dataset(n=500, p=20)
