import string
import clr
from System.Collections.Generic import List
from System import Int32
clr.AddReference("aon/ActivityDiagram.Generator")
from ActivityDiagram.Generator import ArrowGraphGenerator

clr.AddReference("aon/ActivityDiagram.Contracts")
from ActivityDiagram.Contracts.Model.Activities import ActivityDependency
from ActivityDiagram.Contracts.Model.Activities import Activity


def generate_arrow_graph(tasks):
    nodes = tasks[:, 0]
    nodes = [string.ascii_uppercase.index(node.upper()) for node in nodes]
    tasks[:, 0] = nodes

    dependencies_list = tasks[:, 1]
    dependencies_index = [[string.ascii_uppercase.index(dep.upper()) for dep in dependencies if dep != "-"]
                          for dependencies in dependencies_list]

    tasks[:, 1] = dependencies_index

    ActivityDependencies = List[ActivityDependency]()
    for task in tasks:
        #print("Task: {}".format(task[0]))
        Predecessors = List[Int32]()
        for t in task[1]:
            #print("Dependecnies: {}".format(t))
            Predecessors.Add(Int32(t))

        task_id = Int32(task[0])
        ActivityDependencies.Add(ActivityDependency(Activity(task_id), Predecessors))

    arrow_graph_generator = ArrowGraphGenerator(ActivityDependencies)
    activity_arrow_graph = arrow_graph_generator.GenerateGraph()

    edges = []
    # returns Dictionary<ActivityEdge, ActivityEdge>
    ActivityEdges = activity_arrow_graph.Edges
    for activity_edge in ActivityEdges:
        #print("edge [{}, {}]".format(activity_edge.Source.Id, activity_edge.Target.Id))
        #print(activity_edge.Target.Id)
        #print(activity_edge.Activity)
        if activity_edge.Activity:
            edges.append([activity_edge.Source.Id+1, activity_edge.Target.Id+1, {'name': activity_edge.Activity.Id}])
        else:
            edges.append([activity_edge.Source.Id+1, activity_edge.Target.Id+1, {'name': ''}])

    nodes = []
    Vertices = activity_arrow_graph.Vertices
    for verticle in Vertices:
        #print(verticle.Id)
        nodes.append(verticle.Id+1)

    return nodes, edges
