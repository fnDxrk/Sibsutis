class Graph:
    def __init__(self, vertices): # конструктор
        self.V = vertices  # количество вершин
        self.edges = []    # список рёбер

    def add_edge(self, u, v, w):
        self.edges.append((u, v, w))

    def bellman_ford(self, source):
        distance = [float('inf')] * self.V # устанавливаем все расстояния на бесконечность
        distance[source] = 0

        for _ in range(self.V - 1):
            for u, v, w in self.edges:
                if distance[u] != float('inf') and distance[u] + w < distance[v]: # можно ли улучшить расстояние
                    distance[v] = distance[u] + w

        # проверка на наличие циклов отрицательного веса
        for u, v, w in self.edges:
            if distance[u] != float('inf') and distance[u] + w < distance[v]:
                print("Граф содержит цикл отрицательного веса")
                return None

        return distance

g = Graph(4)
g.add_edge(0, 1, 1)
g.add_edge(0, 3, 4)
g.add_edge(1, 2, 2)
g.add_edge(3, 2, 3)
g.add_edge(2, 3, -1)

# g = Graph(5) 
# g.addEdge(0, 1, -1) 
# g.addEdge(0, 2, 4) 
# g.addEdge(1, 2, 3) 
# g.addEdge(1, 3, 2) 
# g.addEdge(1, 4, 2) 
# g.addEdge(3, 2, 5) 
# g.addEdge(3, 1, 1) 
# g.addEdge(4, 3, -3) 

distances = g.bellman_ford(0)

if distances:
    print("Кратчайшие расстояния от вершины 0:", distances)

# 1) Все расстояния устанавливаются в бесконечность, кроме источника, которое равно 0
# 2) Повторяем процесс расслабления всех рёбер V-1 раз (где V — количество вершин)
#    Расслабление означает, что если существует более короткий путь до вершины через другое ребро,
#    мы обновляем расстояние
# 3) Проходя по всем рёбрам, проверяем, можно ли ещё уменьшить расстояние
#    Если это возможно, значит, граф содержит отрицательный цикл.