import os
import subprocess

name = 'cardio_test'

activate_stdout = False
activate_verbose = False
activate_export_html = False
monitor_code_coverage = False
set_threads_number = 0
run_modules = []

test_path = os.getcwd()
stdout = '-s ' if activate_stdout else ''
verbose = '-v ' if activate_verbose else ''
threads = f'-n {set_threads_number} '
run_only = '-k ' + ' and '.join(run_modules) + ' ' if run_modules else ''

test_name = 'unit'
export_html = f'--html={name}_{test_name}_test_report.html --self-contained-html' if activate_export_html else ''
subprocess.call(
    f'python -m pytest '
    f'{verbose} '
    f'{run_only} '
    f'--basetemp=tmp_dir '
    f'{export_html} ' 
    f'{stdout} '
    f'{threads} '
    f'{test_path}',
    shell=True,
)


if __name__ == '__main__':
    import numpy as np

    sig = np.std([1, 2, 1])
    mean = np.mean([1, 2, 1])
    print(sig, mean)

    print((2-mean)/sig)

