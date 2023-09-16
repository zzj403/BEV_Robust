import os


def split_path_string_to_multiname(path_string):
    # 使用os.path.split来拆分路径并创建一个列表
    folder_list = []
    while True:
        head, tail = os.path.split(path_string)
        if not tail:
            break
        folder_list.insert(0, tail)  # 将文件夹名插入到列表的开头
        path_string = head
    return folder_list

if __name__ == '__main__':
    folder_list = split_path_to_multiname("./folder1/folder2/folder3/1.png")
    print(folder_list)
