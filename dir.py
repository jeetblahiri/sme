import os

def print_tree(root, max_items=5, prefix=""):
    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    for i, item in enumerate(items):
        path = os.path.join(root, item)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            # Limit number of displayed items inside each directory
            sub_items = os.listdir(path)
            if len(sub_items) > max_items:
                print_tree_limited(path, max_items, prefix + ("    " if i == len(items)-1 else "│   "))
            else:
                print_tree(path, max_items, prefix + ("    " if i == len(items)-1 else "│   "))


def print_tree_limited(root, max_items, prefix):
    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    for i, item in enumerate(items[:max_items]):
        path = os.path.join(root, item)
        connector = "└── " if i == max_items - 1 else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            print_tree_limited(path, max_items, prefix + ("    " if i == max_items-1 else "│   "))

    if len(items) > max_items:
        print(prefix + "└── ...")


# Usage
root_directory = "./"
print_tree(root_directory)