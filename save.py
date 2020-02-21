from distutils.dir_util import copy_tree

src = str(input("Source : "))
dst = str(input("Destination : "))

copy_tree(src,dst)
