import yaml

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f.read())
icd_version_specified = str(params['prepare_for_xbert']['icd_version'])
diag_or_proc_param = params['prepare_for_xbert']['diag_or_proc']
assert diag_or_proc_param == 'proc' or diag_or_proc_param == 'diag', 'Must specify either \'proc\' or \'diag\'.'
note_category_param = params['prepare_for_xbert']['note_category']
icd_seq_num_param = params['prepare_for_xbert']['one_or_all_icds']


print(type(diag_or_proc_param))
