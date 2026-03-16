import os
import scipy.io as sio

def show_keys(folder, max_files=3):
    print(f'=== Folder: {folder} ===')
    files = [f for f in os.listdir(folder) if f.endswith('.mat')]
    for f in files[:max_files]:
        path = os.path.join(folder, f)
        data = sio.loadmat(path)
        print(f'-- {f}')
        print('  keys:', [k for k in data.keys() if not k.startswith('__')])

if __name__ == '__main__':
    # 这里填上你各个数据集的 mat 路径（高光谱的那个）
    show_keys(r'/home/shiyanshi/dbq/CAVE')
    show_keys(r'/home/shiyanshi/dbq/CST-main/CST-main/dataset/Chikusei_x4')
    show_keys(r'/home/shiyanshi/dbq/Harvard/CZ_hsdb')
    show_keys(r'/home/shiyanshi/dbq/ICVL/test')
    show_keys(r'/home/shiyanshi/dbq/WHU_Hi_HanChuan/test')