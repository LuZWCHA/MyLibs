import os


def gen_txt_file_for(file_name: str, target_dir: str):
    if os.path.exists(target_dir):
        file_list = os.listdir(target_dir)

        file_names = [target_dir + os.sep + name + '\n' for name in file_list]

        with open(file_name, 'w') as f:
            f.writelines(file_names)


if __name__ == '__main__':
    pass
