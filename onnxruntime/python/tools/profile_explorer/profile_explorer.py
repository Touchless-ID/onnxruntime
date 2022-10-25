#!/usr/bin/python

import argparse
import json
import subprocess as sp
import pandas as pd


def _demangle(name, demangler='cu++filt'):
    try:
        with sp.Popen([demangler, name], stdin=sp.PIPE, stdout=sp.PIPE) as proc:
            out, _ = proc.communicate()
            return out.decode('utf-8').strip()
    except:
        return name


def _get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument('input', type=str, help='Trace input file, formatted as JSON')
    parser.add_argument('--demangler', required=False, type=str, default='cu++filt', help='The command to use to demangle C++ identifiers')
    parser.add_argument('--shape-sensitive', action='store_true',
                        help='Perform a shape sensitive analysis of kernel execution times')

    parser.add_argument('--dimension-sensitive', action='store_true',
                        help='Perform a kernel launch dimension sensitive analysis of kernel execution times')

    parser.add_argument('--filter', type=str, nargs='+', action='extend', help='Restrict analysis to the specified identifiers, i.e., specify a filter list')
    parser.add_argument('--csv', help='save data to csv')
    parser.add_argument('-c', '--count', type=int, default=40, help='list top N items')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    return args

def _shape_to_string(shape):
    res = ''
    for dict_obj in shape:
        if len(dict_obj) > 1:
            raise ValueError('Unhandled type in _shape_to_string()')
        key = list(dict_obj.keys())[0]
        value = list(dict_obj.values())[0]
        if len(res) != 0:
            res += '__'
        res += f'{key}_{"x".join(str(v) for v in value)}'
    return res


def _json_to_df(profile_path, filter_set):
    cpu_entries = []
    gpu_entries = []
    
    with open(profile_path, "r", encoding='utf-8') as file_obj:
        data = json.load(file_obj)
    if isinstance(data, dict):
        data = data['traceEvents']

    for item in data:
        cat = item.get('cat')
        if cat is None:
            continue
        dur = item.get('dur')
        if dur is None:
            continue
        arg = item.get('args')
        if arg is None:
            continue
        op_name = arg.get("op_name")

        name = item['name']

        if filter_set is not None and len(filter_set) > 0 and name not in filter_set and op_name not in filter_set:
            continue
        
        if cat != 'Kernel' and not name.endswith('kernel_time'):
            continue

        block_x = arg.get('block_x', -1)
        block_y = arg.get('block_y', -1)
        block_z = arg.get('block_z', -1)
        grid_x = arg.get('grid_x', -1)
        grid_y = arg.get('grid_y', -1)
        grid_z = arg.get('grid_z', -1)

        if cat == 'Kernel':
            gpu_entries.append({
                'name': name,
                'duration': dur,
                'dimensions': f'{block_x}_{block_y}_{block_z}_{grid_x}_{grid_y}_{grid_z}',
                'op_name': op_name,
            })
        else:
            cpu_entries.append({
                'name': item['args']['op_name'],
                'duration': dur,
                'input_shape': _shape_to_string(item['args']['input_type_shape']),
                'output_shape': _shape_to_string(item['args']['output_type_shape']),
            })
                
    cpu_df = pd.DataFrame(cpu_entries)
    gpu_df = pd.DataFrame(gpu_entries)
    cpu_df['count'] = 1
    gpu_df['count'] = 1
    return cpu_df, gpu_df

def _print_cpu_top_hitters(frame, args):
    top = args.count
    group_key = ['name', 'input_shape'] if args.shape_sensitive else ['name']
    frame2 = frame[['duration', 'count']].sum()
    frame['pct'] = 100 * (frame['duration'] / frame2['duration'])
    fields = group_key + ['duration', 'pct', 'count']
    frame1 = frame[fields].groupby(group_key).sum().reset_index()
    frame1 = frame1.sort_values(by='duration', ascending=False)[:top]
    frame1['cumulative_pct'] = frame1['pct'].cumsum()
    frame1['cumulative_dur'] = frame1['duration'].cumsum()
    print('\n------ Top CPU Kernel Times ------')
    print(frame1.round(2).to_string(index=False))
    if args.csv:
        frame1.to_csv(f'{args.csv}_cpu_kernel_times.csv', index=False)

def _print_gpu_top_hitters(frame, args):
    top = args.count
    group_key = ['name', 'dimensions'] if args.dimension_sensitive else ['name']
    frame2 = frame[['duration', 'count']].sum()
    frame['pct'] = 100 * (frame['duration'] / frame2['duration'])
    fields = group_key + ['duration', 'pct', 'count']
    frame1 = frame[fields].groupby(group_key).sum().reset_index()
    frame1 = frame1.sort_values(by='duration', ascending=False)[:top]
    frame1['cumulative_pct'] = frame1['pct'].cumsum()
    frame1['cumulative_dur'] = frame1['duration'].cumsum()
    frame1['name'] = frame1['name'].apply(lambda x: _demangle(x, args.demangler))
    print('\n------ Top GPU Kernel Times ------')
    print(frame1.round(2).to_string(index=False))
    if args.csv:
        frame1.to_csv(f'{args.csv}_gpu_kernel_times.csv', index=False)

def main():
    args = _get_args()
    filter_set = set(args.filter if args.filter is not None else [])

    cpu_df, gpu_df = _json_to_df(args.input, filter_set)

    pd.set_option('display.max_colwidth', 120)
    _print_cpu_top_hitters(cpu_df, args)
    _print_gpu_top_hitters(gpu_df, args)


if __name__ == '__main__':
    main()
