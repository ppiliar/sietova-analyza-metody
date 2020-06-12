# Import the necessary packages
import os
import pathlib
import algo as algo
import string
import numpy as np
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt


from beautifultable import BeautifulTable
from consolemenu import *
from consolemenu.items import *
from consolemenu.prompt_utils import PromptUtils


folder = "input"


def main_menu():
    menu = ConsoleMenu("Mein Kampf", "Vyber si graf:")

    file = [folder, "def.xlsx"]

    default_file = FunctionItem("Pouzit default subor def.xlsl", create_algo_menu, [file])
    custom_file = FunctionItem("Zadat nazov vlastneho suboru", create_algo_menu, [""])


    menu.append_item(default_file)
    menu.append_item(custom_file)

    menu.show()


def create_algo_menu(file):
    if not file:
        file = Screen().input(prompt="Zadaj nazov suboru: ")
        file = [folder, file]
    BOLD = '\033[1m'
    END = '\033[0m'
    algo_menu = ConsoleMenu("Algoritmy. Pouzity subor {}".format(file[1]), "Moznosti:")
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


    algo_menu.append_item(floyd_item)
    algo_menu.append_item(prim_item)
    algo_menu.append_item(graph_item)
    algo_menu.append_item(min_add_item)
    algo_menu.append_item(min_add_max_flow_item)
    algo_menu.append_item(min_add_min_flow_item)
    algo_menu.append_item(e_mat_item)
    algo_menu.append_item(nearest_neighbour_item)
    algo_menu.append_item(best_neighbour_item)
    algo_menu.show()


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
    for res in result:
        print(res)
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
    #print(edges)

    graph.add_edges_from(edges)

    # planar layout works well so far
    # pos = nx.spring_layout(graph, seed=42)
    pos = nx.planar_layout(graph)
    nx.draw(graph, pos)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    # nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")
    nx.draw_networkx_labels(graph, pos)

    dnx_graph = dnx.chimera_graph(2, node_list=nodes, edge_list=edges)
    plt.ion()
    #print(dnx.traveling_salesperson(dnx_graph))

    graph_path = pathlib.Path(folder, "graph.png")
    plt.savefig(graph_path)
    plt.show()
    plt.close()

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

def best_neighbour_matrix(file):
    result = algo.best_neighbour_matrix(file).astype(str)
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
    print("".join(path) + "->{}".format(start.upper()))
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
        table.append_row(row)
    print(table)


if __name__ == "__main__":
    main_menu()
