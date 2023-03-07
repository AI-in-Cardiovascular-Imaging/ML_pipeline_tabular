# import os
#
#
# def checked_dir(dims, strict):
#     """Set dir name according to requested dims"""
#     if not strict:
#         dir_name = 'lenient'
#     elif '2d' in dims and '3d' in dims:
#         dir_name = 'complete'
#     elif '2d' in dims:
#         dir_name = 'complete_2d'
#     elif '3d' in dims:
#         dir_name = 'complete_3d'
#     else:
#         raise NotImplementedError
#
#     return dir_name
