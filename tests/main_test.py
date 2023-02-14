import os
import subprocess

name = 'cardio_test'

activate_stdout = True
activate_verbose = True
activate_export_html = False
run_modules = []

test_path = os.getcwd()
stdout = '-s ' if activate_stdout else ''
verbose = '-v ' if activate_verbose else ''
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
    f'{test_path}',
    shell=True,
)
