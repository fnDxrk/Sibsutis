import heapq

def dijkstra(graph, start):
    
    D = {node: float('inf') for node in graph} # словарь, который будет хранить минимальные расстояния от стартовой вершины до всех остальных вершин, устанавливая их на бесконечность
    D[start] = 0
    S = set()
    priority_queue = [(0, start)] # список, который будет хранить кортежи в формате (расстояние, вершина)

    while priority_queue:
        current_distance, w = heapq.heappop(priority_queue) # извлекаем вершину с минимальным расстоянием current_distance и саму вершину w из очереди
        
        if w in S:
            continue
        
        S.add(w)

        for v, cost in graph[w].items(): # проходим по всем соседям v вершины w и весам рёбер cost
            if v not in S:
                new_distance = current_distance + cost
                if new_distance < D[v]:
                    D[v] = new_distance
                    heapq.heappush(priority_queue, (new_distance, v))

    return D

# Пример использования
graph = {
    0: {1: 4, 2: 1},
    1: {3: 1},
    2: {1: 2, 3: 5},
    3: {}
}

distances = dijkstra(graph, 0)
print(distances) 

#Ищем расстояние от нулевой вершины
# S = {0}
# D[i] = C(0,i)	i = 0 … n
# While S ≠ V do
	# выбираем вершину w, которая принадлежит множеству вершин V\S (V без S) с минимальной стоимостью D(w);
	# S := S + w (добавляем вершину w к множеству S);
	# для всех верши v∈V\S н do  D(v):=min( D(v), D(w) + C(w,v) ) пересчитываем стоимости всех остальных вершин.

