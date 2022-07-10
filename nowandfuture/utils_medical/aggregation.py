import asyncio
import re
import shutil
import multiprocessing
import time
import os
import os.path
from typing import List, Callable, Dict, Any


def func_callback(func1: Callable[[str, str, str, bool, Dict[str, Any]], None], **kwargs):
    func1(**kwargs)


def test(root: str, file_path: str, file_name: str, is_dir: bool, **kwargs):
    target_dir = kwargs.pop("target_dir")
    filter_tuple = kwargs.pop("filter_tuple")
    target = kwargs.pop("target")

    target = file_name if target == "name" else file_path
    o = 0
    time.sleep(0.1)


def consumer_end(root: str, file_path, file_name: str, is_dir: bool, **kwargs):
    target_dir = kwargs.pop("target_dir")
    filter_tuple = kwargs.pop("filter_tuple")
    target = kwargs.pop("target")

    target = file_name if target == "name" else file_path
    if not is_dir:
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        if target.endswith(filter_tuple) and not os.path.exists(target_dir + os.sep + file_name) and os.path.exists(
                target_dir):
            shutil.copyfile(file_path, target_dir + os.sep + file_name)


def consumer_re(root: str, file_path: str, file_name: str, is_dir: bool, **kwargs):
    target_dir = kwargs.pop("target_dir")
    pattern = kwargs.pop("pattern")
    target = kwargs.pop("target")

    target = file_name if target == "name" else file_path
    if not is_dir:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if re.match(pattern, target) and os.path.exists(target_dir):
            shutil.copyfile(file_path, os.path.join(target_dir, file_name))


def consumer_re_all(root: str, file_path: str, file_name: str, is_dir: bool, **kwargs):
    target_dir = kwargs.pop("target_dir")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(target_dir + os.sep + file_name) and os.path.exists(
            target_dir):
        shutil.copyfile(file_path, target_dir + os.sep + file_name)


def consumer_re_dir(root: str, file_path: str, file_name: str, is_dir: bool, **kwargs):
    target_dir = kwargs.pop("target_dir")
    pattern = kwargs.pop("pattern")
    target = kwargs.pop("target")

    target = file_name if target == "name" else file_path

    if is_dir:
        sub_path = file_path[len(root):]
        target_dir = target_dir + sub_path

        if re.match(pattern, target):
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            # shutil.copy(file_path, os.path.split(target_dir)[0])
            copy_files_match_from_to(file_path, target_dir)


def consumer_re_d(root, file_path, file_name: str, is_dir: bool, **kwargs):
    pattern = kwargs.pop("pattern")
    target = kwargs.pop("target")

    target = file_name if target == "name" else file_path

    if re.match(pattern, target) and os.path.exists(file_path):
        print("delete:" + target)
        os.remove(file_path)


def copy_files_from_to(_from: str, _to: str, filter_tuple=(), target="name"):
    Traverse().traverse_files(_from, func=consumer_re_all, target_dir=_to, filter_tuple=filter_tuple, target=target)


def copy_files_end_from_to(_from: str, _to: str, filter_tuple=(), target="name"):
    Traverse().traverse_files(_from, func=consumer_end, target_dir=_to, filter_tuple=filter_tuple, target=target)


def copy_files_match_from_to(_from: str, _to: str, pattern=r'.*', target="name"):
    Traverse().traverse_files(_from, func=consumer_re, target_dir=_to, pattern=pattern, target=target)


def copy_dirs_match_from_to(_from: str, _to: str, pattern=r'.*', target="name"):
    Traverse().traverse_files(_from, func=consumer_re_dir, target_dir=_to, pattern=pattern, target=target)


def delete_files_match_from(_from: str, pattern=r'.*', target="name"):
    Traverse().traverse_files(_from, func=consumer_re_d, target=target, pattern=pattern)


def process_files_match_from_to(_from: str, _to: str, func: Callable[[str, str, str, bool, Dict[str, Any]], None], pattern=r'.*', filter_tuple=(), target='name'):
    Traverse().traverse_files(_from, func=func, target_dir=_to, pattern=pattern, filter_tuple=filter_tuple, target=target)


def process_dirs_match_from_to(_from: str, _to: str, func: Callable[[str, str, str, bool, Dict[str, Any]], None], pattern=r'.*', filter_tuple=(), target='name'):
    Traverse().traverse_files(_from, func=func, target_dir=_to, pattern=pattern, filter_tuple=filter_tuple, target=target)


def group(group_list, func):
    for par in group_list:
        root, file_path, file_name, is_dir, kwargs = par
        func(root, file_path, file_name, is_dir, **kwargs)


def async_process(_from: str, _to: str, func: Callable[[str, str, str, bool, Dict[str, Any]], None], pattern=r'.*', filter_tuple=(), target='name', group_size = 20, pool=None) -> None:
    """
    Process the files async
    :param _from: The files to process
    :param _to: The files to save after processing
    :param func: The callback function, for example: foo(root: str, file_path: str, file_name: str, is_dir: bool, **kwargs)
    :param pattern: The pattern to match
    :param filter_tuple: white filters
    :param target: The file name or the file path to feed to callback function
    :param pool: The process pool
    """

    if not pool:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

    count, par_list = \
        Traverse().traverse_files_async(_from, func=func, target_dir=_to, pattern=pattern, filter_tuple=filter_tuple, target=target)

    start = time.time()
    try:
        i = 0
        group_list = []
        for par in par_list:
            group_list.append(par)
            # root, file_path, file_name, is_dir, kwargs = par
            if i % group_size == group_size - 1:
                pool.apply_async(group, (group_list, func))
                group_list = []
            i += 1

        if group_list:
            pool.apply(group, (group_list, func))

    finally:
        pool.close()
        pool.join()
        print('total time:' + str(time.time() - start))


class Traverse:

    def traverse_files(self, root, work_dir=None, func: Callable[[str, str, str, bool, Dict[str, Any]], None] = None, **kwargs):
        count = 1

        if work_dir is None:
            work_dir = root

        for filename in os.listdir(work_dir):
            file_path = os.path.join(work_dir, filename)
            if os.path.isdir(file_path):
                count += self.traverse_files(root, file_path, func, **kwargs)
                if func:
                    func(root, file_path, filename, True, **kwargs)
            else:
                if func:
                    func(root, file_path, filename, False, **kwargs)
                continue
        return count

    def traverse_files_async(self, root, work_dir=None, func: Callable[[str, str, str, bool, Dict[str, Any]], None] = None, **kwargs):
        count = 1
        func_list = []

        if work_dir is None:
            work_dir = root

        for filename in os.listdir(work_dir):
            file_path = os.path.join(work_dir, filename)
            if os.path.isdir(file_path):
                count += self.traverse_files(root, file_path, func, **kwargs)
                if func:
                    func_list.append([root, file_path, filename, True, kwargs])
            else:
                if func:
                    func_list.append([root, file_path, filename, False, kwargs])
                continue
        return count, func_list


if __name__ == "__main__":
    # path = r'/home/kevin/SharedDatasets/brain/MICCAI_BraTS_2019_Data_Training/LGG'
    # path_to = r'/home/kevin/SharedDatasets/brain/MICCAI_BraTS_2019_Data_Training/HGG_Copy'
    #
    # if not os.path.exists(path_to):
    #     os.makedirs(path_to)
    #
    # # copy_dirs_match_from_to(_from=path, _to=path_to, pattern=r".*\\JPEG", target="path")
    #
    # # for root, dirs, files in os.walk(path):
    # #     for file in files:
    # #         if file.__contains__("cropped"):
    # #             iid = root.replace(path + "\\", '')
    # #             idx = iid.find('\\')
    # #             if idx > 0:
    # #                 iid = iid[8: idx]
    # #             else:
    # #                 iid = iid[8:]
    # #             rename = os.path.join(path_to, iid + '_' + file)
    # #             shutil.copyfile(os.path.join(root, file), rename)
    #
    # # delete_files_match_from(path, pattern=r'.*\\JPEG\\.*png', target="path")
    # async_process(_from=path, _to=path_to, func=test, pattern=r".*", target="path")

    pass
