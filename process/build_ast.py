import json
import os
import re
from collections import OrderedDict
import numpy as np
from graphviz import Digraph
from tree_sitter import Parser, Tree
from tree_sitter import Language

project_name = "ffmpeg/"
task = "test"
C_LANGUAGE = Language('../build/my-languages.so', 'c')


def source_to_ast(source: str):
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    ast_node_obj = parser.parse(source.encode())
    return ast_node_obj


def ast_to_dot(ast_node: Tree):
    def traverse_ast_tree(node_obj, dot_obj):
        dot_obj.node(str(id(node_obj)), node_obj.type)
        for child in node_obj.children:
            dot_obj.edge(str(id(node_obj)), str(id(child)))
            traverse_ast_tree(child, dot_obj)

    dot = Digraph()
    traverse_ast_tree(ast_node.root_node, dot)
    return dot


def edges_to_matrix(edges: list, limit: int):
    matrix = np.full((limit, limit), -1)
    node_max = min(limit, max([max(item[0], item[1]) for item in edges]))
    for u in range(node_max):
        for v in range(node_max):
            matrix[u, v] = -1
    for item in edges:
        if max(item[0], item[1]) < limit:
            matrix[item[0], item[1]] = 1
            matrix[item[1], item[0]] = 1
    return matrix


def merge_matrix(matrix_list: list):
    return np.stack(matrix_list)


def parse_dot_edges(digraph_object):
    edges = []
    discrete_edges = []
    id_set = []
    id_map = {}
    pattern = r"(\d+)\s*->\s*(\d+)"
    for edge in digraph_object.body:
        match = re.search(pattern, edge)
        if match:
            src = match.group(1)
            dst = match.group(2)
            edges.append((src, dst))
            id_set.extend([src, dst])
    id_set = list(OrderedDict.fromkeys(id_set))
    for item in id_set:
        id_map.update({item: len(id_map)})
    for item in edges:
        discrete_edges.append((id_map[item[0]], id_map[item[1]]))
    return discrete_edges


def read_source(json_path: str):
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            data.append(json.loads(line)['func'])
    return data


def write_dot(dot_list: list):
    if not os.path.exists('dots'):
        os.makedirs('dots')
    count = 1
    for item in dot_list:
        with open(f'dots/{count}', 'w') as fs:
            fs.write(item)
        count = count + 1


if __name__ == '__main__':
    # Step1. Get the code snippets from the .jsonl file.
    code_list = read_source(f"../data/raw/{project_name+task}.jsonl")
    # code_list = read_source('data/train-qemu.jsonl')
    # Step2. Convert code snippets to AST objects.
    ast_object_list = []
    for code in code_list:
        ast_object_list.append(source_to_ast(code))
    # Step3. Converting AST objects to Dot objects.
    dot_source_list = []
    dot_object_list = []
    for ast_object in ast_object_list:
        dot_obj = ast_to_dot(ast_object)
        dot_object_list.append(dot_obj)
        dot_source_list.append(dot_obj.source)
    # For easy review, save the transformed dot object locally.
    write_dot(dot_source_list)
    # Step4. Extract edges from Dot objects.
    edges_list = []
    for dot_object in dot_object_list:
        edges_list.append(parse_dot_edges(dot_object))
    # Step5. Convert edges to numpy matrix.
    matrix_list = []
    for edges in edges_list:
        matrix_list.append(edges_to_matrix(edges, 200))
    # Step6. Merge numpy matrices into final output.
    merged_matrix = merge_matrix(matrix_list)
    np.save(f'../data/dataset/{task}_ast.npy', merged_matrix)
    print(merged_matrix.shape)
    print('--- End ---')
