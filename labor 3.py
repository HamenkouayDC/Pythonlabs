import numpy as np
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import time
from tkinter import *
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Алгоритмы ---
def dijkstra(graph, start):
    V = len(graph)
    dist = [np.inf] * V
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v in range(V):
            w = graph[u][v]
            if w != np.inf and dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def floyd_warshall(graph):
    dist = np.copy(graph)
    V = len(graph)
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    if any(dist[i][i] < 0 for i in range(V)):
        raise ValueError("Отрицательный цикл")
    return dist

def bellman_ford(graph, start):
    V = len(graph)
    dist = [np.inf] * V
    dist[start] = 0
    for _ in range(V - 1):
        for u in range(V):
            for v in range(V):
                w = graph[u][v]
                if w != np.inf and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    for u in range(V):
        for v in range(V):
            if graph[u][v] != np.inf and dist[u] + graph[u][v] < dist[v]:
                raise ValueError("Отрицательный цикл")
    return dist

def johnson(graph):
    V = len(graph)
    new_graph = np.full((V+1, V+1), np.inf)
    new_graph[:V, :V] = graph
    for v in range(V):
        new_graph[V][v] = 0

    h = bellman_ford(new_graph, V)
    if any(x == np.inf for x in h):
        raise ValueError("Нельзя применить Джонсона")

    reweighted = np.full_like(graph, np.inf)
    for u in range(V):
        for v in range(V):
            if graph[u][v] != np.inf:
                reweighted[u][v] = graph[u][v] + h[u] - h[v]

    all_dist = []
    for u in range(V):
        d = dijkstra(reweighted, u)
        d = [d[v] - h[u] + h[v] if d[v] != np.inf else np.inf for v in range(V)]
        all_dist.append(d)
    return np.array(all_dist)

def levit(graph, start):
    from collections import deque
    V = len(graph)
    dist = [np.inf] * V
    dist[start] = 0
    M0, M1, M2 = set(), deque([start]), set(range(V))
    M2.remove(start)
    while M1:
        u = M1.popleft()
        for v in range(V):
            w = graph[u][v]
            if w == np.inf or u == v:
                continue
            if v in M2:
                dist[v] = dist[u] + w
                M1.append(v)
                M2.remove(v)
            elif v in M1 and dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
            elif v in M0 and dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                M1.append(v)
                M0.remove(v)
        M0.add(u)
    return dist

# --- Загрузка графа ---
def read_adjacency_matrix(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip()]
    matrix = []
    for line in lines:
        row = []
        for val in line.split():
            row.append(np.inf if val == 'inf' else float(val))
        matrix.append(row)
    return np.array(matrix)

# --- Интерфейс ---
class GraphApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Анализ графов")
        self.graph = None

        self.frame = ttk.Frame(master, padding=10)
        self.frame.grid(row=0, column=0)

        ttk.Label(self.frame, text="Файл графа:").grid(row=0, column=0)
        self.file_entry = ttk.Entry(self.frame, width=30)
        self.file_entry.grid(row=0, column=1)
        ttk.Button(self.frame, text="Загрузить", command=self.load_graph).grid(row=0, column=2)

        ttk.Label(self.frame, text="Начальная вершина:").grid(row=1, column=0)
        self.start_var = IntVar(value=0)
        ttk.Entry(self.frame, textvariable=self.start_var).grid(row=1, column=1)

        ttk.Label(self.frame, text="Конечная вершина:").grid(row=2, column=0)
        self.end_var = IntVar(value=1)
        ttk.Entry(self.frame, textvariable=self.end_var).grid(row=2, column=1)

        ttk.Label(self.frame, text="Алгоритм:").grid(row=3, column=0)
        self.algo_choice = StringVar()
        self.algo_combo = ttk.Combobox(self.frame, textvariable=self.algo_choice, state="readonly")
        self.algo_combo['values'] = ["Дейкстра", "Флойд", "Беллман-Форд", "Левит", "Джонсон"]
        self.algo_combo.current(0)
        self.algo_combo.grid(row=3, column=1)

        ttk.Button(self.frame, text="Найти путь", command=self.find_path).grid(row=4, column=1)

        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=3)

    def load_graph(self):
        try:
            filename = self.file_entry.get()
            self.graph = np.array([
    [0, 2, np.inf, 1],
    [np.inf, 0, 3, np.inf],
    [np.inf, np.inf, 0, 4],
    [np.inf, 1, np.inf, 0]
])

            self.plot_graph()
            messagebox.showinfo("Граф", "Граф успешно загружен.")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))

    def plot_graph(self):
        self.figure.clear()
        G = nx.DiGraph()
        V = len(self.graph)
        for i in range(V):
            for j in range(V):
                if self.graph[i][j] != np.inf:
                    G.add_edge(i, j, weight=self.graph[i][j])
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        self.canvas.draw()

    def find_path(self):
        try:
            if self.graph is None:
                raise ValueError("Сначала загрузите граф")

            start, end = self.start_var.get(), self.end_var.get()
            algo = self.algo_choice.get()

            if algo == "Дейкстра":
                result = dijkstra(self.graph, start)[end]
            elif algo == "Флойд":
                result = floyd_warshall(self.graph)[start][end]
            elif algo == "Беллман-Форд":
                result = bellman_ford(self.graph, start)[end]
            elif algo == "Левит":
                result = levit(self.graph, start)[end]
            elif algo == "Джонсон":
                result = johnson(self.graph)[start][end]
            else:
                result = np.inf

            text = f"{algo}: расстояние от {start} до {end} = {result:.1f}" if result != np.inf else "Путь не найден"
            messagebox.showinfo("Результат", text)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    print("Запуск графического интерфейса...")
    root = Tk()
    app = GraphApp(root)
    root.mainloop()
