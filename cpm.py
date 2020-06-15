
maxCost = 0


class Task:

    def __init__(self, name, cost, dependencies):
        self.name = name
        self.cost = cost
        self.dependencies = dependencies
        self.early_finish = -1
        self.early_start = 0
        self.latest_start = 0
        self.latest_finish = 0
        self.critical_cost = 0

    def __repr__(self):
        critical_cond = "Yes" if self.early_start == self.latest_start else "No"
        to_string = f"{self.name}, {self.early_start}, {self.early_finish}, {self.latest_start}, " \
                    f"{self.latest_finish}, {self.latest_start - self.early_start}, {critical_cond}"
        #to_string = f"{self.name}, {self.cost}"
        return to_string

    def __str__(self):
        critical_cond = "Yes" if self.early_start == self.latest_start else "No"
        to_string = f"{self.name}, {self.early_start}, {self.early_finish}, {self.latest_start}, " \
                    f"{self.latest_finish}, {self.latest_start - self.early_start}, {critical_cond}"
        #to_string = f"{self.name}, {self.cost}"
        return to_string

    def set_latest(self):
        self.latest_start = maxCost - self.critical_cost
        self.latest_finish = self.latest_start + self.cost

    def to_string_array(self):
        critical_cond = "Yes" if self.early_start == self.latest_start else "No"
        to_string = f"{self.name}, {self.early_start}, {self.early_finish}, {self.latest_start}, " \
                    f"{self.latest_finish}, {self.latest_start -self.early_start}, {critical_cond}"
        return to_string

    def is_dependent(self, task):
        if task in self.dependencies:
            return True

        for dep in self.dependencies:
            if dep.is_dependent(task):
                return True
        return False


def critical_path(tasks):
    completed = []
    remaining = tasks.copy()
    while len(remaining) != 0:
        progress = False

        for task in remaining:
            if set(task.dependencies).issubset(completed):
                critical = 0
                for _task in task.dependencies:
                    if _task.critical_cost > critical:
                        critical = _task.critical_cost
                task.critical_cost = critical + task.cost

                completed.append(task)
                remaining.remove(task)
                progress = True

        if not progress:
            raise ValueError("Graf je cyklicky, dalsi vypocet neni mozny")

    max_cost(tasks)
    initial_nodes = initials(tasks)
    calculate_early(initial_nodes)

    ret = completed
    # sort by name
    ret.sort(key=lambda x: x.name)

    return ret


def calculate_early(initials):
    for initial in initials:
        initial.early_start = 0
        initial.early_finish = initial.cost
        set_early(initial)


def set_early(initial):
    completion_time = initial.early_finish
    for task in initial.dependencies:
        if completion_time >= task.early_start:
            task.early_start = completion_time
            task.early_finish = completion_time + task.cost
        set_early(task)


def initials(tasks):
    remaining = tasks.copy()
    for task in tasks:
        for task_dep in task.dependencies:
            if task_dep in remaining:
                remaining.remove(task_dep)


    #print("Initial nodes:")
    #print(remaining)
    return remaining


def max_cost(tasks):
    max = -1
    for task in tasks:
        if task.critical_cost > max:
            max = task.critical_cost

    global maxCost
    maxCost = max

    print("Critical path lenght (cost): {}".format(maxCost))
    for task in tasks:
        task.set_latest()


def print_cpm(tasks):
    for task in tasks:
        print(task.to_string_array())


def get_cpm_table(tasks):
    table = []
    for task in tasks:
        table.append(task.to_string_array().split(','))
    return table


def copm_cpm(actions):
    reversed_deps = actions.copy()

    for task in reversed_deps:
        task[1] = []
        for task_orig in actions:
            for task_dep in task_orig[1]:
                if task_dep is task[0]:
                    task[1].append(task_orig[0])

    task_list = []
    for task in reversed(reversed_deps):
        dep_list = []
        if task[1]:
            for deps in task[1]:
                for x in task_list:
                    if x[0] == deps:
                        dep_list.append(x[1])
        task_list.append([task[0], Task(task[0], int(task[2]), dep_list)])


    task_list = [row[1] for row in task_list]
    result = critical_path(task_list)
    return get_cpm_table(result)

