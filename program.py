# Import the necessary packages
import os
import pathlib
import algo as algo
import cpm as cpm
import string
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import arrowGraphGenerator as agg

from beautifultable import BeautifulTable
from consolemenu import *
from consolemenu.items import *
from consolemenu.prompt_utils import PromptUtils


folder = "input"


def main_menu():
    menu = ConsoleMenu("Mein Kampf", "Vyber si subor s maticou:")

    files = get_menu_files()

    for file in files:
        file_path = [folder, file]
        file_item = FunctionItem("{}".format(file), create_algo_menu, [file_path])
        menu.append_item(file_item)

    menu.show()


def create_algo_menu(file):
    if not file:
        file = Screen().input(prompt="Zadaj nazov suboru: ")
        file = [folder, file]

    algo_menu = ConsoleMenu("Pouzity subor {}".format(file[1]), "Moznosti:")
    floyd_item = FunctionItem("Floydov algo", floyd, [file])
    prim_item = FunctionItem("Minimalna kostra", prim, [file])
    min_add_item = FunctionItem("Minimalny sucet matic", minimal_addition, [file])
    min_add_max_flow_item = FunctionItem("Minimalny sucet matic - maximalna priepustnost",
                                         minimal_addition_maximum_flow, [file])
    min_add_min_flow_item = FunctionItem("Minimalny sucet matic - minimalna priepustnost",
                                         minimal_addition_minimum_flow, [file])
    e_mat_item = FunctionItem("E matica - algoritmus najvyhodnejsieho suseda", best_neighbour_matrix, [file])
    nearest_neighbour_item = FunctionItem("Okruzna cesta - algoritmus najblizsieho suseda",
                                          nearest_neighbour, [file])
    best_neighbour_item = FunctionItem("Okruzna cesta - algoritmus najvyhodnejsieho suseda",
                                          best_neighbour, [file])
    graph_item = FunctionItem("Nakresli graf", create_graph, [file])
    arrow_diagram_item = FunctionItem("Nakresli hranovo-orientovany graf", arrow_diagram, [file])
    node_diagram_item = FunctionItem("Nakresli uzlovo-orientovany graf", node_diagram, [file])
    cpm_table_item = FunctionItem("Cpm tabulka", cpm_table, [file])
    pert_table_item = FunctionItem("Pert tabulka", pert_table, [file])

    algo_menu.append_item(floyd_item)
    algo_menu.append_item(prim_item)
    algo_menu.append_item(graph_item)
    algo_menu.append_item(min_add_item)
    algo_menu.append_item(min_add_max_flow_item)
    algo_menu.append_item(min_add_min_flow_item)
    algo_menu.append_item(e_mat_item)
    algo_menu.append_item(nearest_neighbour_item)
    algo_menu.append_item(best_neighbour_item)
    algo_menu.append_item(arrow_diagram_item)
    algo_menu.append_item(node_diagram_item)
    algo_menu.append_item(cpm_table_item)
    algo_menu.append_item(pert_table_item)
    algo_menu.show()

def get_menu_files():
    files = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))
             and "~$" not in file and ".xlsx" in file]
    return files


def getFileName():
    file = Screen().input(prompt="Zadaj nazov suboru: ")
    floyd(file)


def getFileName1(file, menu):
    file[0] = Screen().input(prompt="Zadaj nazov suboru:")
    menu.show()


def floyd(file):
    try:
        result = algo.floyd(file)
        for key, value in result.items():
            table_print(key, value)

        # print(fl.floyd(file))
    except Exception as e:
        Screen().println("\nError: %s\n" % e)
    finally:
        PromptUtils(Screen()).enter_to_continue()


def prim(file):
    result = algo.prim(file)
    edges = []
    weights = []
    for edge, weight in result:
        v1 = string.ascii_uppercase[edge[0]]
        v2 = string.ascii_uppercase[edge[1]]
        edge = v1 + "->" + v2
        edges.append(edge)
        weights.append(weight)
    print(" , ".join(edges))
    print(" + ".join(map(str, weights)) + " = {}".format(sum(weights)))
    PromptUtils(Screen()).enter_to_continue()


def create_graph(file):
    dist_matrix = algo.get_matrix(file)

    dist_matrix = np.where(dist_matrix == "M", 0, dist_matrix).astype(np.int)

    graph = nx.Graph()

    letters = string.ascii_uppercase
    nodes = [letters[x] for x in range(dist_matrix.shape[0])]

    graph.add_nodes_from(nodes)

    edges = []
    for x in range(dist_matrix.shape[0]):
        for y in range(dist_matrix.shape[1]):
            if dist_matrix[x][y] != 0:
                edges.append([letters[x], letters[y], {'weight': dist_matrix[x][y]}])

    graph.add_edges_from(edges)

    # planar layout works well so far
    # pos = nx.spring_layout(graph, seed=42)
    pos = nx.planar_layout(graph)
    nx.draw(graph, pos)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    # nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")
    nx.draw_networkx_labels(graph, pos)

    graph_path = pathlib.Path(folder, "graph.png")
    fig = plt.savefig(graph_path)
    plt.show()

    plt.close(fig)

    # open file with default image program
    # os.startfile(graph_path)
    PromptUtils(Screen()).enter_to_continue()


def minimal_addition(file):
    result = algo.minimal_addition(file)
    for index, res in enumerate(result):
        table_print("D{}".format(index+1), res)

    PromptUtils(Screen()).enter_to_continue()


def minimal_addition_maximum_flow(file):
    try:
        result = algo.minimal_addition_maximum_flow(file)
        for index, res in enumerate(result):
            table_print("D{}".format(index + 1), res)
    except Exception as e:
        Screen().println("\nError: %s\n" % e)
    finally:
        PromptUtils(Screen()).enter_to_continue()


def minimal_addition_minimum_flow(file):
    try:
        result = algo.minimal_addition_minimum_flow(file)
        for index, res in enumerate(result):
            table_print("D{}".format(index + 1), res)
    except Exception as e:
        Screen().println("\nError: %s\n" % e)
    finally:
        PromptUtils(Screen()).enter_to_continue()


def best_neighbour_matrix(file):
    result = algo.best_neighbour_matrix(file)
    table_print("E", result)
    PromptUtils(Screen()).enter_to_continue()


def nearest_neighbour(file):
    start = Screen().input(prompt="Zadaj startovaci vrchol: ")
    start_index = string.ascii_uppercase.index(start.upper())

    result = algo.nearest_neighbour(file, start_index)
    table_print("", result[1])
    path = [string.ascii_uppercase[node] for node, weight in result[0]]
    weights = [weight for nodes, weight in result[0]]
    print("->".join(path) + "->{}".format(start.upper()))
    print(" + ".join(map(str, weights)) + "= {}".format(sum(weights)))

    PromptUtils(Screen()).enter_to_continue()


def best_neighbour(file):
    start = Screen().input(prompt="Zadaj startovaci vrchol: ")
    start_index = string.ascii_uppercase.index(start.upper())

    result = algo.best_neighbour(file, start_index)
    table_print("", result[1])
    path = [string.ascii_uppercase[node] for node, weight in result[0]]
    weights = [weight for nodes, weight in result[0]]
    print("->".join(path) + "->{}".format(start.upper()))
    print(" + ".join(map(str, weights)) + "= {}".format(sum(weights)))

    PromptUtils(Screen()).enter_to_continue()


def table_print(name, array):
    table = BeautifulTable()
    table.set_style(BeautifulTable.STYLE_BOX)
    # Create array with aplhabet chars
    row_headers = [[string.ascii_uppercase[x] for x in range(array.shape[0])]]
    # Transpose to get collumn with chars
    row_headers = np.asarray(row_headers).T
    # Create array with table name + alphabet chars
    col_headers = [name] + [string.ascii_uppercase[x] for x in range(array.shape[0])]
    # Set column headers
    table.column_headers = col_headers
    # Append first column with aplhabet chars
    array = np.hstack((row_headers, array))

    for row in array:
        print(row)
        print(type(row))
        table.append_row(row)
    print(table)


def arrow_diagram(file):
    tasks = algo.get_tasks(file)
    nodes, edges = agg.generate_arrow_graph(tasks)

    graph = nx.DiGraph()
    letters = string.ascii_uppercase

    dummy_edges = [edge for edge in edges if edge[2]['name'] == '']
    full_edges = [edge for edge in edges if edge[2]['name'] != '']
    # print(full_edges)

    for edge in full_edges:
        edge[2]['name'] = string.ascii_uppercase[int(edge[2]['name'])]

    # Collapse end nodes if possible
    end_nodes = [edge[1] for edge in edges]
    start_nodes = [edge[0] for edge in edges]

    collapse_nodes = [node for node in end_nodes if node not in start_nodes]
    end_nodes = [node for node in end_nodes if node in start_nodes]

    nodes = [node for node in nodes if node not in collapse_nodes[1:]]

    for edge in full_edges:
        if edge[1] in collapse_nodes:
            edge[1] = collapse_nodes[0]

    # Collapse start nodes if possible
    # all end nodes
    end_nodes = [edge[1] for edge in edges]
    # all start nodes
    start_nodes = [edge[0] for edge in edges]

    collapse_nodes = [node for node in start_nodes if node not in end_nodes]
    start_nodes = [node for node in start_nodes if node in end_nodes]

    # remove nodes that can be collapsed but leave one
    nodes = [node for node in nodes if node not in collapse_nodes[1:]]

    for edge in full_edges:
        if edge[0] in collapse_nodes:
            edge[0] = collapse_nodes[0]

    graph.add_nodes_from(nodes)
    graph.add_edges_from(full_edges)
    graph.add_edges_from(dummy_edges)

    # planar layout works well so far
    pos = nx.spring_layout(graph, scale=3)
    #pos = nx.planar_layout(graph, scale=2)
    #pos = nx.kamada_kawai_layout(graph, scale=2)
    fig = plt.figure(1, figsize=(10, 6))
    nx.draw_networkx_edges(graph, pos, full_edges)
    collection = nx.draw_networkx_edges(graph, pos, dummy_edges)
    for patch in collection:
        patch.set_linestyle('dashed')
    nx.draw_networkx_nodes(graph, pos, nodes)
    nx.draw_networkx_labels(graph, pos)
    labels = nx.get_edge_attributes(graph, 'name')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    plt.show()


def node_diagram(file):
    tasks = algo.get_tasks(file)
    nodes, edges = algo.get_tasks_graph(tasks)

    # Add starting node
    end_nodes = [edge[1] for edge in edges]
    # all start nodes
    start_nodes = [edge[0] for edge in edges]

    collapse_nodes = [node for node in start_nodes if node not in end_nodes]
    start_nodes = [node for node in start_nodes if node in end_nodes]

    for edge in edges:
        if edge[0] in collapse_nodes:
            edges.append(['0', edge[0]])

    end_nodes = [edge[1] for edge in edges]
    start_nodes = [edge[0] for edge in edges]

    collapse_nodes = [node for node in end_nodes if node not in start_nodes]
    end_nodes = [node for node in end_nodes if node in start_nodes]

    nodes = np.concatenate((nodes, ['0']))
    nodes = np.concatenate((nodes, ['n+1']))

    for edge in edges:
        if edge[1] in collapse_nodes:
            edges.append([edge[1], 'n+1'])

    graph = nx.DiGraph()

    graph.add_edges_from(edges)
    graph.add_nodes_from(nodes)

    #pos = nx.planar_layout(graph)
    pos = nx.spring_layout(graph, scale=3)
    #pos = nx.kamada_kawai_layout(graph)
    fig = plt.figure(1, figsize=(10, 6))
    nx.draw_networkx_nodes(graph, pos, nodes)
    nx.draw_networkx_edges(graph, pos, edges)
    nx.draw_networkx_labels(graph, pos)

    plt.show()


def cpm_table(file):
    data = cpm.copm_cpm(algo.get_tasks(file))
    table = BeautifulTable()
    table.set_style(BeautifulTable.STYLE_BOX)
    # create table headers
    col_headers = ["Cinnost", "ZM", "KM", "ZP", "KP", "RC", "Kriticka"]

    # Set column headers
    table.column_headers = col_headers

    for row in data:
        table.append_row(row)
    print(table)

    PromptUtils(Screen()).enter_to_continue()


def pert_table(file):
    data, standard_deviation = algo.pert(file)
    table = BeautifulTable()
    table.set_style(BeautifulTable.STYLE_BOX)
    # create table headers
    col_headers = ["Cinnost", "Zavislosti", "aij", "mij", "bij", "trvanie", "odchylka", "rozptyl"]

    # Set column headers
    table.column_headers = col_headers

    for row in data:
        table.append_row(row)
    print(table)
    print("Smerodajna odchylka trvanie projektu q(Te)= {}".format(standard_deviation))

    PromptUtils(Screen()).enter_to_continue()


if __name__ == "__main__":
    main_menu()
