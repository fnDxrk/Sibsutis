import sys
import argparse
import random
import math
from typing import List, Tuple, Optional

# ---
# RSA
# ---

def is_probable_prime(n: int, k: int = 12) -> bool:
    """тест простоты"""
    if n < 2: return False
    small_primes = [2,3,5,7,11,13,17,19,23,29]
    for p in small_primes:
        if n % p == 0: return n == p
    d, s = n-1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n-1: continue
        for __ in range(s-1):
            x = pow(x, 2, n)
            if x == n-1: break
        else:
            return False
    return True

def gen_prime(bits: int) -> int:
    """Генерация простого числа заданной длины в битах"""
    while True:
        p = random.getrandbits(bits) | (1<<(bits-1)) | 1
        if is_probable_prime(p): return p

def egcd(a,b):
    if b==0: return a,1,0
    g,x1,y1 = egcd(b,a%b)
    return g,y1,x1-(a//b)*y1

def modinv(a,m):
    g,x,_ = egcd(a,m)
    if g!=1: raise ValueError("No modular inverse")
    return x % m

def generate_rsa_keypair(bits: int = 256) -> Tuple[int,int,int]:
    """Генерация ключей RSA"""
    while True:
        p,q = gen_prime(bits), gen_prime(bits)
        if p==q: continue
        N = p*q
        phi = (p-1)*(q-1)
        e = 65537
        if math.gcd(e, phi)==1:
            d = modinv(e, phi)
            return e,d,N

# --------------------------
# Работа с графом
# --------------------------

def read_graph_file(path: str) -> Tuple[int, List[Tuple[int,int]], Optional[List[int]]]:
    """Чтение графа и гамильтонова цикла из файла"""
    with open(path,'r',encoding='utf-8') as f:
        tokens = []
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            tokens.extend(line.split())
    n,m = int(tokens[0]), int(tokens[1])
    edges = [(int(tokens[i]),int(tokens[i+1])) for i in range(2,2+2*m,2)]
    cycle = None
    if len(tokens)>2+2*m and len(tokens[2+2*m:])>=n:
        cycle = [int(x) for x in tokens[2+2*m:2+2*m+n]]
    return n,edges,cycle

def build_adj_matrix(n:int, edges:List[Tuple[int,int]]) -> List[List[int]]:
    """Построение матрицы смежности"""
    A = [[0]*n for _ in range(n)]
    for a,b in edges:
        A[a-1][b-1]=1
        A[b-1][a-1]=1
    return A

def find_hamiltonian_cycle_bruteforce(adj:List[List[int]]) -> Optional[List[int]]:
    """Брутфорс для маленьких графов"""
    n=len(adj)
    path=[0]; used=[False]*n; used[0]=True
    def dfs():
        if len(path)==n: return adj[path[-1]][path[0]]==1
        u=path[-1]
        for v in range(n):
            if not used[v] and adj[u][v]==1:
                used[v]=True; path.append(v)
                if dfs(): return True
                path.pop(); used[v]=False
        return False
    if dfs(): return [v+1 for v in path]
    return None

# ------------
# Перестановка
# ------------

def permute_graph(adj:List[List[int]]) -> Tuple[List[List[int]],List[int]]:
    """Перестановка графа (H=perm(G))"""
    n=len(adj)
    perm=list(range(n))
    random.shuffle(perm)
    H=[[adj[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
    return H, perm

def commit_matrix(H:List[List[int]], e:int,N:int) -> Tuple[List[List[int]],List[List[int]],List[List[int]]]:
    """Коммитменты RSA"""
    n=len(H)
    F,s_mat,r_mat=[[0]*n for _ in range(n)],[[0]*n for _ in range(n)],[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            r=random.randrange(1,max(2,N-1))
            s=H[i][j]+2*r
            F[i][j]=pow(s,e,N)
            s_mat[i][j]=s
            r_mat[i][j]=r
    return F,s_mat,r_mat

def apply_permutation_to_cycle(cycle:List[int], perm:List[int]) -> List[int]:
    """Применение перестановки к циклу"""
    n=len(perm)
    pos_in_H=[-1]*n
    for h_idx,g_idx in enumerate(perm): pos_in_H[g_idx]=h_idx
    return [pos_in_H[v-1]+1 for v in cycle]

# --------------------------
# Проверка Бобом
# --------------------------

def verify_ciphertext(s:int,e:int,N:int,c_expected:int)->bool:
    return pow(s,e,N)==c_expected

def verify_revealed_cycle_on_H(edges:List[Tuple[int,int]],n:int)->bool:
    """Проверка, что раскрытые ребра образуют гамильтонов цикл"""
    if len(edges)!=n: return False
    deg=[[0]*n for _ in range(n)]; d=[0]*n
    for u,v in edges: d[u]+=1; d[v]+=1; deg[u][v]=deg[v][u]=1
    if any(x!=2 for x in d): return False
    visited=[False]*n; stack=[0]; visited[0]=True
    while stack:
        x=stack.pop()
        for y in range(n):
            if deg[x][y] and not visited[y]: visited[y]=True; stack.append(y)
    return all(visited)

# ------------
# Демонстрация
# ------------

def print_graph(adj:List[List[int]],name="G"):
    print(f"\nГраф {name} (матрица смежности):")
    for row in adj: print(" ".join(str(x) for x in row))

def print_permutation(perm:List[int]):
    print("Перестановка: ", ", ".join(str(x+1) for x in perm))

# --------
# Протокол
# --------

def run_protocol_once(adj_G:List[List[int]],cycle_G:List[int],e:int,d:int,N:int)->bool:
    H,perm=permute_graph(adj_G)
    F,s_mat,r_mat=commit_matrix(H,e,N)
    challenge=random.choice([1,2])
    if challenge==1:
        cycle_in_H=apply_permutation_to_cycle(cycle_G,perm)
        edges=[(cycle_in_H[i]-1,cycle_in_H[(i+1)%len(cycle_in_H)]-1) for i in range(len(cycle_in_H))]
        print("Bob запросил: показать цикл (1)")
        print("Гамильтонов цикл в H:", " ".join(str(x) for x in cycle_in_H))
        print("Рёбра цикла H:", ", ".join(f"({u+1},{v+1})" for u,v in edges))
        ok=True
        for u,v in edges:
            if not verify_ciphertext(s_mat[u][v],e,N,F[u][v]): ok=False; break
        if ok and not verify_revealed_cycle_on_H(edges,len(H)): ok=False
    else:
        print("Bob запросил: показать изоморфизм (2)")
        print_permutation(perm)
        ok=True
        # проверка соответствия H и G через perm
        for i in range(len(H)):
            for j in range(len(H)):
                if H[i][j]!=adj_G[perm[i]][perm[j]]: ok=False; break
            if not ok: break
    print("Bob принимает ответ:", ok)
    return ok

# --------------
# Главная логика
# --------------

def protocol_demo(graph_file:str,cycle_file:Optional[str],rounds:int=8,rsa_bits:int=256):
    n,edges,cycle_in_file=read_graph_file(graph_file)
    if cycle_file:
        with open(cycle_file,'r',encoding='utf-8') as f:
            tokens=[x for line in f if line.strip() and not line.startswith('#') for x in line.split()]
        cycle=[int(x) for x in tokens[:n]]
    else:
        cycle=cycle_in_file

    A=build_adj_matrix(n,edges)
    print_graph(A,"G")
    print("Гамильтонов цикл в G (1-indexed):", " ".join(str(x) for x in cycle))

    e,d,N=generate_rsa_keypair(bits=rsa_bits)

    successes=0
    for r in range(1,rounds+1):
        print(f"\n--- РАУНД {r} ---")
        ok=run_protocol_once(A,cycle,e,d,N)
        if ok: successes+=1
    print(f"\nРезультат: Боб принял {successes} из {rounds} раундов.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("graph")
    parser.add_argument("--cycle",default=None)
    parser.add_argument("--rounds",type=int,default=8)
    parser.add_argument("--rsa-bits",type=int,default=256)
    args=parser.parse_args()
    protocol_demo(args.graph,args.cycle,args.rounds,args.rsa_bits)

if __name__=="__main__":
    main()
