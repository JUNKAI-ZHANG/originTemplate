> #   										**Zjkai**





# Graph 图论

## 图

### 图的存储方式

#### vector存图

``` c++
const int maxn = 2e5 + 7;

vector<int > g[maxn];
vector<vector<int> > g(maxn);
vector<vector<pair<int, int> > > g(maxn);
```

#### 链式前向星

* 慎用，可能会很慢；
* 可以在网络流相关的题使用，快速得到与本条边反向的边（i ^ 1);

``` c++
struct LSQXX {
	struct Edge {
		int ne, to, fl;
		ll w;

		Edge(int ne_ = 0, int to_ = 0, ll w_ = 0, int fl_ = 0) : ne(ne_), to(to_), w(w_), fl(fl_) {

		}

		bool operator<(const Edge& a) const {
			return w < a.w;
		};
	};
	int n, ptr;
	vector<int> head;
	vector<Edge> e;

	LSQXX (int n_ = 0) : n(n_), head(n + 1, -1){ 
		ptr = -1;
	} 

	void add(int fr, int to, ll w, int fl = 0) {
		e[++ptr] = Edge(head[fr], to, w, fl);
		head[fr] = ptr;
	}

	void Sort() {
		sort(e.begin(), e.begin() + ptr);
	}
	
};
```

#### 存边

```c++
const int maxn = 5e6 + 7;

struct EDGE {
	struct Edge {
		int fr, to, fl;
		ll w;

		Edge(int fr_ = 0, int to_ = 0, ll w_ = 0, int fl_ = 0) : fr(fr_), to(to_), w(w_), fl(fl_) {

		}

		bool operator<(const Edge& a) const {
			return w < a.w;
		};
	};
	int n, ptr;
	vector<Edge> e;

	EDGE (int n_ = 0) : n(n_), e(maxn) { 
		ptr = 0;
	} 

	void add(int fr, int to, ll w, int fl = 0) {
		e[++ptr] = Edge(fr, to, w, fl);
	}
	
	void Sort() {
		sort(e.begin() + 1, e.begin() + 1 + ptr);
	}
	
};
```

#### 分层图

##### 适用场景

* 一些图论题，比如最短路、网络流等，题目对边的权值提供可选的操作，比如可以将一定数量的边权减半，在此基础上求解最优解。

##### 算法思路

> 根据是否进行题目提供的操作以及操作次数的不同，会产生非常多的情况，如果考虑何时使用操作，情况更是多。如果将在图上求解最短路看成是在二维平面上进行的，引入进行操作的次数 k 做为第三维，那么这个三维空间就理应可以包含所有的情况，便可以在这个三维空间上解决问题。每进行一次操作（k+1），除了操作的边，其他边没有任何变化，在 k=0,1,2,…，时图都是一样的，那么就将图复制成 k+1 份，第 i 层图代表进行了 i 次操作后的图。每相邻两层图之间的联系，应该决定了一次操作是发生在哪条边上（何时进行操作）。根据操作的特点（对边权的修改）可以 i 层点到 i+1 层点的边来表示一次操作。

![img](https://img-blog.csdnimg.cn/20181217151240943.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNzM2MDM2,size_16,color_FFFFFF,t_70)

## 最短路

### 堆优化的Dijkstra

```c++
template<typename T>
struct Graph {
	int n;
	vector<vector<pair<int, int> > > g;
	Graph(int _n = 0) : n(_n), g(_n + 1) {}

	void read(int m) {
		for(int i = 1;i <= m;i++) {
			int u, v, w;
			cin >> u >> v >> w;
			g[u].push_back(pair<int, int>(v, w));
		}
	}

	T dijkstra(int st, int ed) {
		vector<T> dis(n + 1, 0x3f3f3f3f);
		priority_queue<pair<T, int>, vector<pair<T, int> >, greater<pair<T, int> > > q;
		q.push(pair<T, int>(0, st));
		dis[st] = 0;
		while(q.size()) {
			pair<T, int> now = q.top();
			q.pop();
			int x = now.second;
			for(pair<int, T> it : g[x]) {
				int to = it.first;
				T w = it.second;
				if(dis[to] > dis[x] + w) {
					dis[to] = dis[x] + w;
					q.push(pair<T, int>(dis[to], to));
				}
			}
		}
		return dis[ed];
	}

};
```

### 	*SPFA*

> SPFA(Shortest Path Faster Algorithm)算法是求单源最短路径的一种算法，它是Bellman-ford的队列优化，它是一种十分高效的最短路算法。
>
> 很多时候，给定的图存在负权边，这时类似Dijkstra等算法便没有了用武之地，而Bellman-Ford算法的复杂度又过高，SPFA算法便派上用场了。SPFA的复杂度大约是O(kE),k是每个点的平均进队次数(一般的，k是一个常数，在稀疏图中小于2)。
>
> 但是，SPFA算法稳定性较差，在稠密图中SPFA算法时间复杂度会退化。
>
> 实现方法：建立一个队列，初始时队列里只有起始点，标记起点的vis为1，在建立一个dis数组记录起始点到所有点的最短路径（该表格的初始值要赋为极大值，该点到他本身的路径赋为0）。然后执行松弛操作，用队列里有的点去刷新起始点到所有点的最短路，如果刷新成功且被刷新点不在队列中则把该点加入到队列最后。重复执行直到队列为空。
>
> 此外，SPFA算法还可以判断图中是否有负权环，即一个点入队次数超过N。

#### Floyd

求最短路

```c++
	vector<vector<ll> > a(n + 1, vector<ll>(n + 1, 0x3f3f3f3f3f3f3f));
	vector<vector<ll> > dp(n + 1, vector<ll>(n + 1, 0x3f3f3f3f3f3f3f));
	
	for(int i = 1;i <= m;i++) {
		int u, v, w;
		cin >> u >> v >> w;
		a[u][v] = w;
		a[v][u] = w;
	}

	dp = a;

	ll ans = 0x3f3f3f3f3f3f3f;//最小环路值
	for(int k = 1;k <= n;k++) {
        
		for(int i = 1;i < k;i++) {
			for(int j = i + 1;j < k;j++) {
				ans = min(ans, a[i][j] + dp[j][k] + dp[k][i]);  //更新最小环路
			}
		}
        
		for(int i = 1;i <= n;i++) {
			for(int j = 1;j <= n;j++) {
				a[i][j] = min(a[i][j], a[i][k] + a[k][j]);
			}
		}
	}
}
```

## LCA

### 倍增Lca

```c++
const int maxn = 1e5 + 4;
struct Edge {
	int to;
	int ne;
}edge[maxn << 1];
int ptr = 0;
vector<int > head(maxn, -1), dep(maxn, 0);
vector<vector<int> > st(maxn, vector<int>(25, 0));
int n, m;
void init() {
	ptr = 0;
	head = vector<int>(maxn, -1);
	st = vector<vector<int> >(maxn, vector<int>(25, 0));
	dep = vector<int>(maxn, 0);
}
void add(int st, int end) {
	ptr++;
	edge[ptr].to = end;
	edge[ptr].ne = head[st];
	head[st] = ptr;
	ptr++;
	edge[ptr].to = st;
	edge[ptr].ne = head[end];
	head[end] = ptr;
}
void dfs(int now, int fa) {
	dep[now] = dep[fa] + 1;
	st[now][0] = fa;
	for(int i = 1;(1 << i) <= dep[now];i++) {
		st[now][i] = st[st[now][i - 1]][i - 1];
	}
	for(int i = head[now]; i != -1;i = edge[i].ne) {
		int nex = edge[i].to;
		if(nex == fa) continue;
		st[nex][0] = now;
		dfs(nex, now);
	}
}
int lca(int x, int y) {
	if(dep[x] < dep[y]) swap(x, y);
	for(int i = 14;i >= 0;i--){
		if(dep[st[x][i]] >= dep[y]) {
			x = st[x][i];
		}
		if(x == y) return x;
	}
	for(int i = 14;i >= 0;i--) {
		if(st[x][i] != st[y][i]) {
			x = st[x][i];
			y = st[y][i];
		}
	}
	return x == y?x : st[x][0];
}
```

### 树剖Lca

```c++
const int maxn = 2e5 + 7;

vector<vector<int> > g(maxn);
vector<int> sz(maxn), dep(maxn), son(maxn);
vector<int> top(maxn), fa(maxn);

void dfs_1(int x) {
	sz[x] = 1, dep[x] = dep[fa[x]] + 1;
	for(int to : g[x]) {
		if(to == fa[x]) continue;
		fa[to] = x;
		dfs_1(to);
		sz[x] += sz[to];	
		if(sz[to] > sz[son[x]]) son[x] = to;
	}
	return ;
}

void dfs_2(int x, int Tp) {
	top[x] = Tp;
	if(son[x]) dfs_2(son[x], Tp);
	for(int to : g[x]) {
		if(to == fa[x] || to == son[x]) continue;
		dfs_2(to, to);
	}
	return ;
}

int lca(int x, int y) {
	while(top[x] != top[y]) {
		if(dep[top[x]] < dep[top[y]]) y = fa[top[y]];
		else x = fa[top[x]];
	}
	return dep[x] < dep[y]? x : y;
}


```

## Tarjan

### 缩点

```c++
const int maxn = 1e5 + 7;

vector<vector<int> > g(maxn), g_(maxn);
vector<int> dfn(maxn), low(maxn), scc(maxn), size(maxn << 1);
stack<int> st;

int Time, cnt;

void tarjan(int x) {
	low[x] = dfn[x] = ++Time;
	st.push(x);
	vis[x] = 1;
	for(int to : g[x]) {
		if(dfn[to] == 0) {
			tarjan(to);
			low[x] = min(low[x], low[to]);
		}
		else if(vis[to]) {
			low[x] = min(low[x], dfn[to]);
		}
	}
	if(low[x] == dfn[x]) {
		++cnt;
		while(1) {
			int tp = st.top();
			vis[tp] = 0;
			scc[tp] = cnt;
			size[cnt]++;
			st.pop();
			if(tp == x) break;
		}
	}
}
```

### 割点

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

const int maxn = 1e5 + 7;

vector<vector<int> > g(maxn);
vector<int> ans;

int dfn[maxn], vis[maxn], low[maxn];

void dfs(int x, int fa, int dep) {
	low[x] = dfn[x] = dep;
	vis[x] = 1;
	int child = 0;
	for(int to : g[x]) {
		if(to != fa) {
			if(vis[to] == 1) {
				low[x] = min(low[x], dfn[to]);
			}
		}
		if(vis[to] == 0) {
			dfs(to, x, dep + 1);
			low[x] = min(low[x], low[to]);
			child++;
			if((fa == -1 && child > 1) || (fa != -1 && low[to] >= dfn[x])) ans.push_back(x);
		}
	}
	vis[x] = 2;
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	int n, m;
	cin >> n >> m;
	for(int i = 1;i <= m;i++) {
		int u, v;
		cin >> u >> v;
		g[u].push_back(v);
		g[v].push_back(u);
	}
	for(int i = 1;i <= n;i++) if(!dfn[i]) dfs(i, -1, 1);
	sort(ans.begin(), ans.end());
	ans.erase(unique(ans.begin(), ans.end()), ans.end());
	cout << ans.size() << '\n';
	for(int i : ans) cout << i << ' ';


	return 0;
}
```

### 割边

```cpp
#include <iostream>
#include <string.h>
#include <string>
#include <algorithm>
#include <math.h>
#include <vector>
using namespace std;
const int maxn = 123456;
int n, m, dfn[maxn], low[maxn], vis[maxn], ans, tim;
bool cut[maxn];
vector<int> edge[maxn];
void cut_bri(int cur, int pop) {
	vis[cur] = 1;
	dfn[cur] = low[cur] = ++tim;
	int children = 0; ////子树 
	for (int i : edge[cur]) {////对于每一条边 
		if (i == pop || vis[cur] == 2) 
			continue;
		if (vis[i] == 1) ////遇到回边 
			low[cur] = min(low[cur], dfn[i]); ////回边处的更新 (有环)
		if (vis[i] == 0) {
			cut_play(i, cur);
			children++;  ////记录子树数目 
			low[cur] = min(low[cur], low[i]); ////父子节点处的回溯更新
			if ((pop == -1 && children > 1) || (pop != -1 && low[i] >= dfn[cur])) {////判断割点 
				if (!cut[cur])
					ans++;		 ////记录割点个数
				cut[cur] = true; ///处理割点
			}
			if(low[i]>dfn[cur]) {////判断割边 
				bridge[cur][i]=bridge[i][cur]=true;  ////low[i]>dfn[cur]即说明(i,cur)是桥(割边)； 
			} 
		}
	}
	vis[cur] = 2; ////标记已访问 
}
int main() {
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= m; i++) {
		int x, y;
		scanf("%d%d", &x, &y);
		edge[x].push_back(y);
		edge[y].push_back(x);
	}
	for (int i = 1; i <= n; i++) {
		if (vis[i] == 0)
			cut_bri(i, -1); ////防止原来的图并不是一个连通块 
			////对于每个连通块调用一次cut_bri 
	}
	printf("%d\n", ans);
	for (int i = 1; i <= n; i++) ////输出割点 
		if (cut[i])
			printf("%d ", i);
	return 0;
}
```

## 虚树

```c++
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

const int maxn = 250007;
const int inf = 1e9;
vector<int> RG[maxn],VG[maxn];
int U[maxn],V[maxn],C[maxn];
int dfn[maxn],deep[maxn];ll me[maxn];int fa[maxn][20];
int stk[maxn],top;
int n,m,idx;
void dfs(int x){
	dfn[x]= ++idx;
	deep[x] = deep[fa[x][0]] + 1;
	for(int to : RG[x]){
		if(to == fa[x][0]) continue;
		fa[to][0] = x;
		dfs(to);
	}
}

int LCA(int u,int v){
	if(deep[u] < deep[v]) swap(u,v);
	int delta = deep[u] - deep[v];
	for(int i = 19;i >= 0;--i){
		if((delta >> i) & 1) u = fa[u][i];
	}
	for(int i = 19;i >= 0;--i){
		if(fa[u][i] != fa[v][i]) u = fa[u][i],v = fa[v][i];
	}
	if(u == v) return u;
	return fa[u][0];
}

bool comp(int a,int b){
	return dfn[a] < dfn[b];
}

void insert(int u){
	if(top == 1) {stk[++top] = u;return;}
	int lca = LCA(u,stk[top]);
	if(lca == stk[top]) {stk[++top] = u;return ;}
	while(top > 1 && dfn[lca] <= dfn[stk[top-1]]){
		VG[stk[top-1]].push_back(stk[top]);
		VG[stk[top]].push_back(stk[top-1]);
		--top;
	}
	if(lca != stk[top]) {
		VG[lca].push_back(stk[top]);
		VG[stk[top]].push_back(lca);
		stk[top] = lca;
	} 
	stk[++top] = u;
}

int idq[maxn],mark[maxn];

void DP(int x, int fa){
	printf("%d ", x);
	for(int to : VG[x]) {
		if(to == fa) continue;
		DP(to, x);
	}
}

int main(){
	ios::sync_with_stdio(false);
	cin >> n;
	int sz;
	cin >> sz;
	for(int i = 1;i < n;++i){
		cin >> U[i] >> V[i];
		RG[U[i]].push_back(V[i]);
		RG[V[i]].push_back(U[i]);
	}
	dfs(1);
	for(int t = 1;t <= 19;++t) for(int i = 1;i <= n;++i){
		fa[i][t] = fa[fa[i][t-1]][t-1];
	}
	for(int j = 0;j < sz;++j){
		cin >> idq[j];
		mark[idq[j]] = 1;
	}
	sort(idq,idq+sz,comp);
	top = 0;
	stk[++top] = 1;
	for(int j = 0;j < sz;++j) insert(idq[j]);
	while(top > 0) {
		VG[stk[top-1]].push_back(stk[top]);
		top--;
	}
	DP(idq[0], 0);
	for(int j = 0;j < sz;++j) VG[idq[j]].clear(),mark[idq[j]] = 0;
	VG[0].clear();


	return 0;
}
```

## 	最小生成树

### Kruskal

```c++
const int maxn = 5e5 + 7;
vector<int> fa(maxn, 0);
int n, m;
struct Edge {
	int fr, to, w;
	bool operator<(const Edge& b) const {
		return w < b.w;
	}
}edge[maxn];
int find(int x) {
	return x == fa[x]? x : fa[x] = find(fa[x]);
}
void init() {   // 一定要调用
	cin >> n >> m;
	for(int i = 1;i <= n;i++) fa[i] = i;
	for(int i = 1;i <= m;i++) cin >> edge[i].fr >> edge[i].to >> edge[i].w;
	sort(edge + 1, edge + 1 + m);
}
void work(){  //core！
	init();
	ll cnt = 0, mst = 0;
	for(int i = 1;i <= m;i++) {
		int fx = find(edge[i].fr);
		int fy = find(edge[i].to);
		if(fx != fy) {
			cnt++;
			mst += edge[i].w;
			fa[fx] = fy;
		}
		if(cnt == n - 1) break;
	}
    //不联通图没有最小生成树，注意
	cout << mst << '\n';
}
```

### Prim

``` c++
const int maxn = 5e3 + 5;

vector<vector<int> > g(maxn, vector<int>(maxn, 0x3f3f3f3f));

int Prim(int n) {
	vector<int> dis(n + 1, 0x3f3f3f3f);
	vector<int> vis(n + 1, 0);
	int mst = 0;
	for(int i = 1;i <= n;i++) dis[i] = g[1][i];
	vis[1] = 1;
	for(int i = 2;i <= n;i++) {
		int id = -1, minEdge = 0x3f3f3f3f;
		for(int j = 1;j <= n;j++) {
			if(!vis[j] && minEdge > dis[j]) {
				minEdge = dis[j];
				id = j;
			}
		}
		vis[id] = 1;
		mst += minEdge;
		for(int j = 1;j <= n;j++) {
			if(vis[j] == 0 && dis[j] > g[id][j]) {
				dis[j] = g[id][j];
			}
		}
	}
	return mst;
}
```

# Network 网络流

## 二分图

### 二分图的判断

* 二染色法：图可以被分为两个集合，每条边的两个端点都可以被划分到两个集合中；这两个集合可以被染成两个颜色
* 偶环法：根据二分图的定义，二分图中不可能存在奇数长度的环

```c++
//偶环法

#include<bits/stdc++.h>

using namespace std;
using ll = long long;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

	int t;
	cin >> t;
	while(t--) {
		int n, m, ok = 1;
		cin >> n >> m;

		vector<vector<int> > g(n + 1);
		vector<int> vis(n + 1, 0), dep(n + 1, 0);

		for(int i = 1;i <= m;i++) {
			int u, v;
			cin >> u >> v;
			g[u].emplace_back(v);
			g[v].emplace_back(u);
		}

		function<void(int, int, int)> dfs = [&](int x, int fa, int depp) {
			vis[x] = 1;
			dep[x] = depp;
			for(int to : g[x]) {
				if(to == fa) continue;
				if(vis[to] == 1) {
					if((dep[x] - dep[to]) % 2 == 0) {
						ok = 0;
					}
					continue;
				}
				dfs(to, x, depp + 1);
			}
		};

		for(int i = 1;i <= n && ok;i++) if(vis[i] == 0) dfs(i, 0, 0);

		if(ok) cout << "YES" << '\n';
		else cout << "NO" << '\n';
	}

    return 0;
}
```

### 二分图最大匹配

* 匈牙利算法（O(V * E)）

```c++
int M, N;            //M, N分别表示左、右侧集合的元素数量
int Map[MAXM][MAXN]; //邻接矩阵存图
int p[MAXN];         //记录当前右侧元素所对应的左侧元素
bool vis[MAXN];      //记录右侧元素是否已被访问过
bool match(int i) {
    for (int j = 1; j <= N; ++j)
        if (Map[i][j] && !vis[j]) {//有边且未访问
            vis[j] = true;                 //记录状态为访问过
            if (p[j] == 0 || match(p[j])) {//如果暂无匹配，或者原来匹配的左侧元素可以找到新的匹配
                p[j] = i;    //当前左侧元素成为当前右侧元素的新匹配
                return true; //返回匹配成功
            }
        }
    return false; //循环结束，仍未找到匹配，返回匹配失败
}
int Hungarian() {
    int cnt = 0;
    for (int i = 1; i <= M; ++i) {
        memset(vis, 0, sizeof(vis)); //重置vis数组
        if (match(i))
            cnt++;
    }
    return cnt;
}
```

### 二分图带权最大匹配(KM算法)

* 当权值为1时求得的匹配为二分图的最大匹配(O(3))

```c++
const int maxn = 105;
const int inf = 0x3f3f3f3f;

int n, m, t, nx, ny;//Attention: nx, ny
vector<int> match(maxn, 0), lx(maxn, 0), ly(maxn, 0);
vector<int> slack(maxn, 0), vis_x(maxn, 0), vis_y(maxn, 0);
vector<vector<int> > g(maxn, vector<int>(maxn, 0));

bool dfs(int x) {
	vis_x[x] = 1;
	for (int y = 1; y <= ny; y++) {
		if (vis_y[y]) continue;
		int tmp = lx[x] + ly[y] - g[x][y];
		if (tmp == 0) {
			vis_y[y] = 1;
			if (match[y] == -1 || dfs(match[y])) {
				match[y] = x;
				return true;
			}
		} 
		else if (slack[y] > tmp) {
			slack[y] = tmp;
		}
	}
	return false;
}
int KM() {
	match = vector<int>(maxn, -1);
	ly = vector<int>(maxn, 0);
	int res = 0;
	for (int i = 1; i <= nx; i++) {
		lx[i] = -inf;
		for (int j = 1; j <= ny; j++) {
			if (g[i][j] > lx[i]) {
				lx[i] = g[i][j];
			}
		}
	}
	for (int x = 1; x <= nx; x++) {
		for (int i = 1; i <= ny; i++) slack[i] = inf;
		while (true) {
			vis_x = vector<int>(maxn, 0);
			vis_y = vector<int>(maxn, 0);
			if (dfs(x)) break;
			int d = inf;
			for (int i = 1; i <= ny; i++) if (!vis_y[i] && d > slack[i]) d = slack[i];
			for (int i = 1; i <= nx; i++) if (vis_x[i]) lx[i] -= d;

			for (int i = 1; i <= ny; i++) {
				if (vis_y[i]) ly[i] += d;
				else slack[i] -= d;
			}
		}
	}
	for (int i = 1; i <= ny; i++) {
		if (match[i] != -1) {
			res += g[match[i]][i];
		}
	}
	return res;
}
```

## Flow

### Dinic求最大流

> 层次网络中，汇点的层次就代表着源点到汇点的最短距离，在每次构建层次网络时，汇点的层次必然会比上一次至少多1，因为在每次的dfs中，所有的最短路径都被找了出来，并且经过流量调整后当前所有的最短路径均被阻塞，因此在下一次构建层次网络时，没有办法再找到这么短的路径了，也就是说源点到汇点的最短距离至少增加1，也就是汇点的层次至少加1。而层次的上界显然为顶点的个数，因此最外层的while循环至多遍历O(V)次。内层的bfs需要O(E)的时间，dfs为找所有的增广路，由对EK算法的分析不难得出，找到所有的增广路不会超过O(VE)的时间，因此Dinic算法的复杂度上界为O(V^2E)，这个性能要优于EK算法，而且Dinic算法的实现也很简便，我认为是解题的首选算法。

```c++
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

const int inf = 0x7fffffff;
const int maxn = 1e5 + 7;

struct Edge {
	int to, w, ne;
	Edge() {}
	Edge(int a, int b, int c) {
		to = a, w = b, ne = c;
	}
}edge[maxn << 2];

vector<int> head(maxn, -1), dep(maxn, -1);
int tot = -1, n, m, s, t;

void addd(int fr, int to, int w) {
	edge[++tot] = Edge(to, w, head[fr]);
	head[fr] = tot;
}

void add(int fr, int to, int w) {
	addd(fr, to, w), addd(to, fr, 0);
}

int dfs(int u, int flow) {
	if(u == t) return flow;
	int ret = 0;
	for(int i = head[u]; i != -1; i = edge[i].ne) {
		int to = edge[i].to;
		if(dep[to] == dep[u] + 1 && edge[i].w > 0) {
			int fl = dfs(to, min(flow, edge[i].w));
			flow -= fl, ret += fl;
			edge[i].w -= fl, edge[i ^ 1].w += fl;
			if (!flow) break;
		}
	}
	if(!ret) dep[u] = -1;
	return ret;
}

bool bfs() {
	dep = vector<int>(maxn, -1);
	queue<int> q; 
	q.push(s);
	dep[s] = 0;
	while(q.size()) {
		int now = q.front();
		q.pop();
		for(int i = head[now]; i != -1; i = edge[i].ne) {
			int to = edge[i].to;
			if(dep[to] == -1 && edge[i].w > 0) {
				dep[to] = dep[now] + 1;
				q.push(to);
			}
		}
	}
	return dep[t] != -1;
}

ll dinic() {
	ll max_flow = 0;
	while(bfs()) {
		max_flow += dfs(s, inf);
	}
	return max_flow;
}

int main() {
	scanf("%d %d %d %d", &n, &m, &s, &t);//起点必须从1开始
	for(int i = 0; i < m; i++) {
		int u, v, w;
		scanf("%d %d %d", &u, &v, &w);
		add(u, v, w);
	}

	cout << dinic() << '\n';
	return 0;
}
```

### ISAP求最大流

> Sap算法是对Dinic算法一个小的优化，在Dinic算法中，每次都要进行一次bfs来更新层次网络，这未免有些过于浪费，因为有些点的层次实际上是不需要更新的。Sap算法就采取一边找增广路，一边更新层次网络的策略。注意在Sap算法中源点的层次应该是最高的，一定要有Gap优化，不然这个算法的性能就不尽如人意了。Sap算法的复杂度上界和Dinic一样也是O(V^2E)

```C++
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

const int inf = 1 << 30;
const int N = 1e6 + 40;

int idx = 1, n, m, s, t;
int dep[N << 1], gap[N << 1], head[N << 1];

struct edge {
    int to;
    int next;
    int val;
} e[N << 1];

void add(int u, int v, int d) {
    idx++;
    e[idx].to = v;
    e[idx].val = d;
    e[idx].next = head[u];
    head[u] = idx;
}

void bfs() {
    memset(dep, -1, sizeof(dep));
    memset(gap, 0, sizeof(gap));
    dep[t] = 0, gap[0] = 1;
    queue<int>q;
    q.push(t);

    while (q.size()) {
        int u = q.front();
        q.pop();

        for (int i = head[u]; i; i = e[i].next) {
            int v = e[i].to;

            if (dep[v] != -1)
                continue;

            q.push(v);
            dep[v] = dep[u] + 1;
            gap[dep[v]]++;
        }
    }

    return;
}

ll maxflow;

int dfs(int u, int flow) {
    if (u == t) {
        maxflow += flow;
        return flow;
    }

    int used = 0;

    for (int i = head[u]; i; i = e[i].next) {
        int d = e[i].to;

        if (e[i].val && dep[d] + 1 == dep[u]) {
            int minn = dfs(d, min(e[i].val, flow - used));

            if (minn) {
                e[i].val -= minn;
                e[i ^ 1].val += minn;
                used += minn;
            }

            if (used == flow)
                return used;
        }
    }

    -- gap[dep[u]];

    if (gap[dep[u]] == 0)
        dep[s] = n + 1;

    dep[u] ++;
    gap[dep[u]] ++;
    return used;
}
ll ISAP() {
    maxflow = 0;
    bfs();

    while (dep[s] <= n)
        dfs(s, inf);

    return maxflow;
}
int main() {
    scanf("%d%d%d%d", &n, &m, &s, &t);

    for (int i = 1; i <= m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        add(u, v, w);
        add(v, u, 0);
    }

    printf("%lld\n", ISAP());

    return 0;
}
```

### HLPP求最大流

> HLPP算法即最高标号预流推进算法，它的特点时并不采取找增广路的思想，而是不断地在可行流中找到那些仍旧有盈余的节点，将其盈余的流量推到周围可接纳流量的节点中，具体什么意思呢？对于一个最大流而言，除了源点和汇点以外所有的其他节点都应该满足流入的总流量等于流出的总流量，如果首先让源点的流量都尽可能都流到其相邻的节点中，这个时候相邻的节点就有了盈余，即它流入的流量比流出的流量多，所以要想办法将这些流量流出去。这种想法其实很自然，如果不知道最大流求解的任何一种算法，要手算最大流的时候，采取的策略肯定会是这样，将能流的先流出去，遇到容量不足的边就将流量减少，直到所有流量都流到了汇点。
> 但是这样做肯定会遇到一个问题，会不会有流量从一个节点流出去然后又流回到这个节点？如果这个节点是源点的话这么做是没问题的，因为有的时候通过某些节点是到达不了汇点的，这个时候要将流量流回到源点，但是其他情况就可能会造成循环流动，因此需要用到层次网络，只在相邻层次间流动。
>
> > **HLPP(Highest Label Preflow Push)最高标签预流推进算法**是处理网络最大流里两种常用方法——**增广路**&**预流推进**中，预流推进算法的一种。其他科学家证明了其复杂度是**紧却的O(n^2 · sqrt(m))**。在随机数据中不逊色于普通的增广路算法，而在精心构造的数据中无法被卡，所以是一种可以替代Dinic的方法（随我怎么说，代码又长又难调，所以还是Dinic好啊）
> >
> > 但无论怎样，wikiwiki里面已经承认HLPP是现在最优秀的网络流算法了。

> 那么**预流推进**这个大门类里面，思想都差不多。大抵上就是我们对每个点记录**超额流(Extra FlowExtra Flow)** ，即**允许流在非源点暂时存储**，并**伺机将超额流推送出去**。不可推送的，就会流回源点。那么最终答案显然存储在Extra[T]里面。但同时这也有一个问题，就是会出现两个点相互推送不停的情况。为了防止这样，我们采用**最高标号**的策略，给每个点一个高度，对于一个点uu以及它的伴点集合{v}，当且仅当hu=hv+1 时才可以推送流。并且我们对于源点S，设置hS=N，并对于S实行**无限制推送**。那么最后的答案就保存在Extra[T]里面 。但有时，我们发现有个点是”谷“，即周围点的高度都比它高，但是它有超额流。那么我们此时考虑**拔高它的高度**，即**重贴标签(relabel)**操作。

```c++
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

const int inf = 0x3f3f3f3f;
const int maxn = 2e3 + 5;
const ll INF = 0x3f3f3f3f3f3f3f3fll;

struct HLPP {
    struct Edge {
        int v, rev;
        ll cap;
    };
    int n, sp, tp, lim, ht, lcnt;
    ll exf[maxn];
    vector<Edge> G[maxn];
    vector<int> hq[maxn], gap[maxn], h, sum;
    void init(int nn, int s, int t) {
        sp = s, tp = t, n = nn, lim = n + 1, ht = lcnt = 0;

        for (int i = 1; i <= n; ++i)
            G[i].clear(), exf[i] = 0;
    }
    void add_edge(int u, int v, ll cap) {
        G[u].push_back({v, int(G[v].size()), cap});
        G[v].push_back({u, int(G[u].size()) - 1, 0});
    }
    void update(int u, int nh) {
        ++lcnt;

        if (h[u] != lim)
            --sum[h[u]];

        h[u] = nh;

        if (nh == lim)
            return;

        ++sum[ht = nh];
        gap[nh].push_back(u);

        if (exf[u] > 0)
            hq[nh].push_back(u);
    }
    void relabel() {
        queue<int> q;

        for (int i = 0; i <= lim; ++i)
            hq[i].clear(), gap[i].clear();

        h.assign(lim, lim), sum.assign(lim, 0), q.push(tp);
        lcnt = ht = h[tp] = 0;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (Edge &e : G[u])
                if (h[e.v] == lim && G[e.v][e.rev].cap)
                    update(e.v, h[u] + 1), q.push(e.v);

            ht = h[u];
        }
    }
    void push(int u, Edge &e) {
        if (!exf[e.v])
            hq[h[e.v]].push_back(e.v);

        ll df = min(exf[u], e.cap);
        e.cap -= df, G[e.v][e.rev].cap += df;
        exf[u] -= df, exf[e.v] += df;
    }
    void discharge(int u) {
        int nh = lim;

        if (h[u] == lim)
            return;

        for (Edge &e : G[u]) {
            if (!e.cap)
                continue;

            if (h[u] == h[e.v] + 1) {
                push(u, e);

                if (exf[u] <= 0)
                    return;
            } else if (nh > h[e.v] + 1)
                nh = h[e.v] + 1;
        }

        if (sum[h[u]] > 1)
            update(u, nh);
        else {
            for (; ht >= h[u]; gap[ht--].clear())
                for (int &i : gap[ht])
                    update(i, lim);
        }
    }
    ll hlpp() {
        exf[sp] = INF, exf[tp] = -INF, relabel();

        for (Edge &e : G[sp])
            push(sp, e);

        for (; ~ht; --ht) {
            while (!hq[ht].empty()) {
                int u = hq[ht].back();
                hq[ht].pop_back();
                discharge(u);

                if (lcnt > (n << 2))
                    relabel();
            }
        }

        return exf[tp] + INF;
    }
} hp;
signed main() {
    int n, m, s, t, u, v, w;
    scanf("%d %d %d %d", &n, &m, &s, &t);
    hp.init(n, s, t);   //总点数，起点，终点

    for(int i = 1;i <= m;i++) {
        scanf("%d %d %d", &u, &v, &w);
        hp.add_edge(u, v, w);
    }

    cout << hp.hlpp() << '\n';
    return 0;
}
```

### 最小费用最大流(SPFA）

```c++
#include<bits/stdc++.h>

using namespace std;
using ll = long long;

const int maxm = 5e5 + 7;
const int maxn = 1e3 + 6;
const int inf  = 0x7f7f7f7f;

vector<int > head(maxm, -1);
vector<int > pre(maxn, 0), cost(maxn, inf), flow(maxn, inf), vis(maxn, 0), last(maxn, 0);
queue<int> q;

struct Edge {
	int ne, to, cost, v;
	//cost: 
	//v: capacity
	Edge() {}
	Edge(int a, int b, int c, int d) : ne(a), to(b), cost(c), v(d) {}
}edge[maxm];
int tot = -1, n, m;//第一条边的编号必须为0

void add(int fr, int to, int v, int cost) {
	edge[++tot] = Edge(head[fr], to, cost, v);
	head[fr] = tot;
}

bool spfa(int st, int ed) {
	cost = flow = vector<int>(n + 1, inf);
	q.push(st);
	vis[st] = 1, cost[st] = 0, pre[ed] = 0;
	while(q.size()) {
		int now = q.front();
		q.pop();
		vis[now] = 0;
		for(int i = head[now];i != -1;i = edge[i].ne) {
			int to = edge[i].to;
			if(cost[to] > cost[now] + edge[i].cost && edge[i].v > 0) {
				cost[to] = cost[now] + edge[i].cost;
				flow[to] = min(flow[now], edge[i].v);
				pre[to] = now;
				last[to] = i;
				if(vis[to] == 0) {
					vis[to] = 1;
					q.push(to);
				}
			}
		}
	}
	return pre[ed];
}

void mcmf() {
	int maxFlow = 0;
	int minCost = 0;

	while(spfa(1, n)) {
		int tmp = n;
		maxFlow += flow[n];
		minCost += flow[n] * cost[n];

		while(tmp != 1) {
			edge[last[tmp]].v -= flow[n];
			edge[last[tmp] ^ 1].v += flow[n];
			tmp = pre[tmp];
		}

	}

	cout << maxFlow << ' ' << minCost << '\n';

}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	cin >> n >> m;
	for(int i = 1;i <= m;i++) {
		int u, v, c, w;
		cin >> u >> v >> c >> w;
		add(u, v, c, w);
		add(v, u, 0, -w);
	}

	mcmf();

    return 0;
}
```

# Data Structure 数据结构

## 并查集

```c++
struct DSU {
	int n;
	vector<int> fa, rank;
	DSU(int n_ = 0) : n(n_), fa(n_ + 1), rank(n_ + 1) {
		iota(fa.begin(), fa.end(), 0);
	}

	int find(int x) {
		return x == fa[x] ? x : fa[x] = find(fa[x]);
	}

	void merge(int x, int y) {
		x = find(x), y = find(y);
		if(x == y) return ;
		if(rank[x] < rank[y]) fa[x] = y;
		else {
			fa[y] = x;
			if(rank[x] == rank[y]) rank[x]++;
		}
	}
};
```

## 线段树

### 加法线段树

```c++
template<typename T>
struct segTree {
	int n;
	vector<T> tr, tag;
	segTree(int n_ = 0) : n(n_), tr(n_ << 2), tag(n_ << 2) {}

	void up(int rt) {
		return ;
	}

	void down(int rt, int l, int r) {
		if(tag[rt]) {
			int mid = l + r >> 1;//lson -> [l, mid], rson -> [mid + 1, r]
		}
		return ;
	}

	void build(int rt, int l, int r){
		if(l == r){
			return ;
		}
		int mid = l + r >> 1;
		build(rt << 1, l, mid);
		build(rt << 1 | 1, mid + 1, r);
		up(rt);
		return ;
	}

	void modify(int rt, int l, int r, int val, int L, int R) {
		if(l <= L && R <= r) {
			return ;
		}
		down(rt, L, R);
		int mid = L + R >> 1;
		if(mid >= l) modify(rt << 1, l, r, val, L, mid);
		if(mid <  r) modify(rt << 1 | 1, l, r, val, mid + 1, R);
		up(rt);
		return ;
	}

	T query(int rt, int l, int r, int L, int R){
		if(l <= L && R <= r) return tr[rt];
		down(rt, L, R);
		int mid = L + R >> 1;
		T ret{};
		if(l <= mid) ret += query(rt << 1, l, r, L, mid);
		if(r >  mid) ret += query(rt << 1 | 1, l, r, mid + 1, R);
		return ret;
	}
};
```

### 乘法线段树

```c++
const int p = 1e9 + 7;

struct segTree {
	int n;
	vector<ll> tr, f, mf;
	segTree(int n_ = 0) : n(n_), tr(n_ << 2, 0), f(n_ << 2, 0), mf(n_ << 2, 1) {
        
    }

	void up(int rt) {
		tr[rt] = (tr[rt << 1] + tr[rt << 1 | 1]) % p;
		return ;
	}

	void down(int rt, int l, int r) {
		int mid = l + r >> 1;
		tr[rt << 1] = (tr[rt << 1] * mf[rt] % p + f[rt] * (mid - l + 1)) % p;
		tr[rt << 1 | 1] = (tr[rt << 1 | 1] * mf[rt] % p + f[rt] * (r - mid)) % p;
		mf[rt << 1] = mf[rt << 1] * mf[rt] % p;
		mf[rt << 1 | 1] = mf[rt << 1 | 1] * mf[rt] % p;
		f[rt << 1] = (f[rt << 1] * mf[rt] + f[rt]) % p;
		f[rt << 1 | 1] = (f[rt << 1 | 1] * mf[rt] + f[rt]) % p;
		f[rt] = 0, mf[rt] = 1;
		return ;
	}

	void build(int rt, int l, int r){
		if(l == r){
			cin >> tr[rt];
			return ;
		}
		int mid = l + r >> 1;
		build(rt << 1, l, mid);
		build(rt << 1 | 1, mid + 1, r);
		up(rt);
		return ;
	}

	void modify(int rt, int l, int r, ll v, ll mv, int L, int R) {
		if(l <= L && R <= r) {
			mf[rt] = mf[rt] * mv % p;
			f[rt] = (f[rt] * mv + v) % p;
			tr[rt] = (tr[rt] * mv + v * (R - L + 1) % p) % p;
			return ;
		}
		down(rt, L, R);
		int mid = L + R >> 1;
		if(mid >= l) modify(rt << 1, l, r, v, mv, L, mid);
		if(mid <  r) modify(rt << 1 | 1, l, r, v, mv, mid + 1, R);
		up(rt);
		return ;
	}

	ll query(int rt, int l, int r, int L, int R){
		if(l <= L && R <= r) return tr[rt];
		down(rt, L, R);
		int mid = L + R >> 1;
		ll ret = 0;
		if(l <= mid) ret = (ret + query(rt << 1, l, r, L, mid)) % p;
		if(r >  mid) ret = (ret + query(rt << 1 | 1, l, r, mid + 1, R)) % p;
		return ret;
	}
};
```

### 最值线段树

```c++
struct segTree {
	int n;
	vector<int> min_, f;
	segTree(int n_ = 0) : n(n_), min_(n_ << 2, 0), f(n_ << 2, 0) {}

	inline void up(int rt) {
		min_[rt] = min(min_[rt << 1], min_[rt << 1 | 1]);
	}

	inline void down(int rt, int l, int r) {
		if(f[rt]) {
			f[rt << 1] += f[rt];
			f[rt << 1 | 1] += f[rt];
			min_[rt << 1] -= f[rt];
			min_[rt << 1 | 1] -= f[rt];
			f[rt] = 0;
		}
		return ;
	}

	void build(int rt, int l, int r) {
		if(l == r) {
			cin >> min_[rt];
			return ;
		}
		int mid = l + r >> 1;
		build(rt << 1, l, mid);
		build(rt << 1 | 1, mid + 1, r);
		up(rt);
		return ;
	}

	void modify(int rt, int l, int r, int val, int L, int R) {
		if(l <= L && R <= r) {
			min_[rt] -= val;
			f[rt] += val;
			return ;
		}
		down(rt, L, R);
		int mid = L + R >> 1;
		if(mid >= l) modify(rt << 1, l, r, val, L, mid);
		if(mid <  r) modify(rt << 1 | 1, l, r, val, mid + 1, R);
		up(rt);
		return ;
	}

	int query(int rt, int l, int r, int L, int R) {
		if(l <= L && R <= r) return min_[rt];
		down(rt, L, R);
		int mid = L + R >> 1;
		int ret = 0x3f3f3f3f;
		if(l <= mid) ret = min(ret, query(rt << 1, l, r, L, mid));
		if(r >  mid) ret = min(ret, query(rt << 1 | 1, l, r, mid + 1, R));
		up(rt);
		return ret;
	}
};
```

### 动态开点线段树

> 普通的线段树在内存上通常为数组大小的四倍，但是当数组范围达到$10^9$的时候，普通的线段树就不再适用，因为查询最多不会超过$10^6$次，所涉及到的点也就没有达到$10^9$这个数量级，所以我们可以使用动态开点线段树，对内存进行优化，单条树链的长度最多为$log(n)$，所以时间复杂度和空间复杂度均为$O(qlogn)$,达到时空平衡的状态。

* node：记录点的信息（当前点的值，当前点的懒标记，左孩子和右孩子的指针）
  * up() ：上传操作，记得先将自己的值清空，再加上左右孩子的值（如果不是空指针）
  * down() ：懒标记下传
* modify()：区间加法，[L, R]记录当前递归到了哪个区间，[l, r]记录查询的区间，若当前区间是要查询区间的子区间，则直接进行区间加法，否则继续向下递归；
* query() ：区间查询，[L, R]记录当前递归到了哪个区间，[l, r]记录查询的区间，若当前区间是要查询区间的子区间，则直接返回当前区间的值，否则继续向下递归；

```c++
template<typename T>
struct DynamicSegtree {
	struct node {
		T data, tag;
		node *lc, *rc;
		node(T data_ = 0) : data(data_), tag(0), lc(nullptr), rc(nullptr) {}
		void up() {
			data = 0;
			if(lc != nullptr) data += lc -> data;
			if(rc != nullptr) data += rc -> data;
			return ;
		}
		void down(int l, int r) {
			if(lc == nullptr) lc = new node;
			if(rc == nullptr) rc = new node;
			int mid = l + r >> 1;
			lc -> data += (mid - l + 1) * tag;
			lc -> tag += tag;
			rc -> data += (r - mid) * tag;
			rc -> tag += tag;
			tag = 0;
		}
	}*rt = new node(0);
	void modify(node* rt, int l, int r, T val, int L, int R) {
		if(l <= L && R <= r) {
			rt -> data += (R - L + 1) * val;
			rt -> tag += val;
			return ;
		}
		rt -> down(L, R);
		int mid = L + R >> 1;
		if(l <= mid) modify(rt -> lc, l, r, val, L, mid);
		if(r > mid)  modify(rt -> rc, l, r, val, mid + 1, R);
		rt -> up();
		return ;
	}
	T query(node* rt, int l, int r, int L, int R) {
		if(l <= L && R <= r) {
			return rt -> data;
		}
		rt -> down(L, R);
		int mid = L + R >> 1;
		T ret{};
		if(l <= mid) ret += query(rt -> lc, l, r, L, mid);
		if(r > mid)  ret += query(rt -> rc, l, r, mid + 1, R);
		return ret;
	}
};
```

## 树状数组

### 一维树状数组

#### 正常的树状数组（单点更新，区间求和）

```c++
template<typename T>
struct Fenwick {
	int n;
	vector<T> tr;
	Fenwick(int n_ = 0) : n(n_), tr(n_ + 1) {}
	void add(int x, int v_) {
		if(!x) { tr[x] += v_; return ; }
		while(x <= n) tr[x] += v_, x += x & -x;
	}
	T query(int x) {
		T ans(tr[0]);
		while(x) ans += tr[x], x -= x & -x;
		return ans;
	}
};
```

### 二维树状数组

```c++
struct Fenwick {
	int n, m;
	vector<vector<int> > tr;
	Fenwick() {}
	Fenwick(int n_, int m_) : n(n_), m(m_), tr(n << 1, vector<int>(m_ << 1, 0)) {}
	void add(int xx, int yy, int val) {
		for(int x = xx;x <= n;x += x & -x) {
			for(int y = yy;y <= m;y += y & -y) {
				tr[x][y] += val;
			}
		}
	}
	int query(int xx, int yy){
		int ret = 0;
		for(int x = xx;x > 0;x -= x & -x) {
			for(int y = yy;y > 0;y -= y & -y) {
				ret += tr[x][y];
			}
		}
		return ret;
	}
};
```

## RMQ(倍增法)

### 	ST表(前置芝士)

$$
f[j][i]<==>j\to{j+2^i}
$$

```c++
struct SparseTable { // a_ : [1, n]
	int n;
	vector<vector<int> > St;
	vector<int> lg;
	SparseTable(const vector<int>& a_, int n_) : n(n_), St(n_ + 1, vector<int>(20, 0)), lg(n_ + 1) {
		for(int i = 2;i <= n;i++) lg[i] = lg[i >> 1] + 1;
		for(int i = 1;i <= n;i++) St[i][0] = a_[i];
		for(int j = 1;j <= lg[n];j++) {
			for(int i = 1;i + (1 << j) - 1 <= n;i++) {
				St[i][j] = __gcd(St[i][j - 1], St[i + (1 << (j - 1))][j - 1]);
			}
		}
	}
	int query(int l, int r) {
		int x = lg[r - l + 1];
		return __gcd(St[l][x], St[r - (1 << x) + 1][x]);
	}
};
```

## 树链剖分

### 重链剖分

``` c++
const int maxn = 2e5 + 7;

struct segTree {
	//@author  Zjkai
	int l, r;
	ll val, f;
}tr[maxn << 2];

struct Edge {
	int ne, to, w;
}edge[maxn];

vector<int> head(maxn, -1);
vector<int> fa(maxn, 0), dep(maxn, 0), son(maxn, 0), size(maxn, 0); //dfs_1
// 每个点的父亲//深度数组//重儿子数组//子树大小数组
vector<int> L(maxn, 0), invL(maxn, 0), top(maxn, 0);                //dfs_2
// DFN // DFN中对应的点//重链顶
vector<int> pi(maxn, 0);
//点权
int tot = 0, n, m, r, p, Time = 0;

void add(int fr, int to, int cost) {
	++tot;
	edge[tot].ne = head[fr];
	edge[tot].to = to;
	edge[tot].w = cost;
	head[fr] = tot;
}

void up(int rt) {
	tr[rt].val = tr[rt << 1].val + tr[rt << 1 | 1].val;
}

void down(int rt){
	if(tr[rt].f) {
		tr[rt << 1].val += tr[rt].f * (tr[rt << 1].r - tr[rt << 1].l + 1);
		tr[rt << 1 | 1].val += tr[rt].f * (tr[rt << 1 | 1].r - tr[rt << 1 | 1].l + 1);
		tr[rt << 1].f += tr[rt].f;
		tr[rt << 1 | 1].f += tr[rt].f;
		tr[rt].f = 0;
	}
	return ;
}

void build(int rt, int l, int r){
	tr[rt].l = l, tr[rt].r = r, tr[rt].f = 0;
	if(l == r){
		tr[rt].val = pi[invL[l]];
		return ;
	}
	int mid = l + r >> 1;
	build(rt << 1, l, mid);
	build(rt << 1 | 1, mid + 1, r);
    up(rt);
	return ;
}

void modify(int rt, int l, int r, int val) {//修改子树
	if(tr[rt].l >= l && tr[rt].r <= r) {
		tr[rt].val += (tr[rt].r - tr[rt].l + 1) * val;
		tr[rt].f += val;
		return ;
	}
	down(rt);
	int mid = tr[rt].l + tr[rt].r >> 1;
	if(mid >= l) modify(rt << 1, l, r, val);
	if(mid <  r) modify(rt << 1 | 1, l, r, val);
	up(rt);
	return ;
}

ll query(int rt, int l, int r){//查询子树
	if(tr[rt].l >= l && tr[rt].r <= r) return tr[rt].val;
	down(rt);
	int mid = tr[rt].l + tr[rt].r >> 1;
	ll ret = 0;
	if(l <= mid) ret += query(rt << 1, l, r);
	if(r >  mid) ret += query(rt << 1 | 1, l, r);
	return ret;
}

void dfs_1(int now) {//预处理1
	size[now] = 1;
	dep[now] = dep[fa[now]] + 1;
	for(int i = head[now];i != -1;i = edge[i].ne) {
		int to = edge[i].to;
		if(to != fa[now]) {
			fa[to] = now;
			dfs_1(to);
			size[now] += size[to];
			if(size[to] > size[son[now]]) son[now] = to;
		}
	}
	return ;
}

void dfs_2(int now, int tp) {// 预处理2
	L[now] = ++Time;
	invL[Time] = now;
	top[now] = tp;
	if(son[now]) dfs_2(son[now], tp);

	for(int i = head[now];i != -1;i = edge[i].ne) {
		int to = edge[i].to;
		if(to != fa[now] && to != son[now]) {
			dfs_2(to, to);   //轻链的顶就是自己!
		}
	}
	return ;
}

ll getsum(int x, int y) {//得到简单路径的长度
	ll ret = 0;
	while(top[x] != top[y]) {
		if(dep[top[x]] < dep[top[y]]) swap(x, y);
		ret = ret + query(1, L[top[x]], L[x]);
		x = fa[top[x]];
	}
	if(L[x] > L[y]) swap(x, y);
	return ret + query(1, L[x], L[y]);
}

void upd(int x, int y, int val) {//修改简单路径
	while(top[x] != top[y]) {
		if(dep[top[x]] < dep[top[y]]) swap(x, y);
		modify(1, L[top[x]], L[x], val);
		x = fa[top[x]];
	}
	if(L[x] > L[y]) swap(x, y);
	modify(1, L[x], L[y], val);
}
```

## 分块|根号分治

### 分块

```c++
template<typename T>
struct Sqrt_DivideConquer {
	int n, len;
	vector<T> a, belong, L, R, tag, sum;
	Sqrt_DivideConquer(int n_ = 0, const vector<T> &a_ = {}) : belong(n_ + 1){
		n = n_, a = a_, len = sqrt(n_);
		L.resize(len << 1), R.resize(len << 1);
		tag.resize(len << 1), sum.resize(len << 1);
		for(int i = 1;i <= len + 1;i++) {
			L[i] = n / len * (i - 1) + 1;
			R[i] = min(n / len * i, n_);
			for(int j = L[i];j <= R[i];j++) {
				belong[j] = i;
				sum[i] += a_[j];
			}
		}
	}
	void add(int l, int r, T val) {
		if(belong[l] == belong[r]) {
			for(int i = l;i <= r;i++) {
				a[i] += val;
				sum[belong[i]] += val;
			}
		}
		else {
			for(int i = l;i <= R[belong[l]];i++) {
				a[i] += val;
				sum[belong[i]] += val;
			}
			for(int i = L[belong[r]];i <= r;i++) {
				a[i] += val;
				sum[belong[i]] += val;
			}
			for(int i = belong[l] + 1;i < belong[r];i++) {
				tag[i] += val;
			}
		}
		return ;
	}

	T query(int l, int r) {
		T ret{};
		if(belong[l] == belong[r]) {
			for(int i = l;i <= r;i++) {
				ret += a[i] + tag[belong[i]];
			}
		}
		else {
			for(int i = l;i <= R[belong[l]];i++) {
				ret += a[i] + tag[belong[i]];
			}
			for(int i = L[belong[r]];i <= r;i++) {
				ret += a[i] + tag[belong[i]];
			}
			for(int i = belong[l] + 1;i < belong[r];i++) {
				ret += (R[i] - L[i] + 1) * tag[i] + sum[i];
			}
		}
		return ret;
	}
};
```

### 莫队

* 处理区间问题**暴力首选**

```c++
#include<bits/stdc++.h>

using namespace std;
using ll = long long;

const int maxn = 1e5 + 7;

struct Off {
	int l, r;
	int id;
};

void add(int x) {

}

void sub(int x) {

}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	int n, m, res = 0;
	cin >> n >> m;
	res = 0;
	vector<int> a(n + 1, 0), cnt(maxn, 0), ans(m + 1, 0);
	vector<int> bl(n + 1, 0);//块的编号
	vector<Off> q(m + 1);    //离线
	int size_ = (int)(sqrt(n));

	for(int i = 1;i <= n;i++) {
		cin >> a[i];
		cnt[a[i]]++;
		bl[i] = i / size_;
	}

	for(int i = 1;i <= m;i++) {
		cin >> q[i].l >> q[i].r;
		q[i].id = i;
	}

	sort(q.begin() + 1, q.end(), [&](Off x, Off y) {
			if(bl[x.l] == bl[y.l]) return x.r < y.r;
			return bl[x.l] < bl[y.l];
	});

	int l = 1, r = 0;
	for(int i = 1;i <= m;i++) {
		q[i].l += 1, q[i].r -= 1;
		while(q[i].l < l) sub(--l);
		while(q[i].r > r) sub(++r);
		while(q[i].l > l) add(l++);
		while(q[i].r < r) add(r--);
		ans[q[i].id] = res;
	}

	for(int i = 1;i <= m;i++) cout << ans[i] << '\n';

	return 0;
}
```

## 堆

### FHQ-Treap(无旋树堆)

```c++
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

const int maxn = 1e5 + 7;

mt19937 rng(chrono::system_clock::now().time_since_epoch().count());

struct Node {
	int val;
	int l, r, size, Priority;
}tr[maxn << 2];

int tot, Root, n, x, opt;

int create(int key) {
	int root = ++tot;
	tr[root].val = key;
	tr[root].size = 1;
	tr[root].l = tr[root].r = 0;
	tr[root].Priority = rng();
	return root;
}

void pushup(int root) {
	tr[root].size = tr[tr[root].l].size + tr[tr[root].r].size + 1;
}

void split(int root, int key, int &x, int &y) {
	if (root == 0) {
		x = y = 0;
		return;
	}
	if (tr[root].val <= key) {
		x = root;
		split(tr[root].r, key, tr[root].r, y);
	} 
	else {
		y = root;
		split(tr[root].l, key, x, tr[root].l);
	}
	pushup(root);
}

int merge(int x, int y) {
	if (x == 0 || y == 0) return x + y;
	if (tr[x].Priority > tr[y].Priority) {
		tr[x].r = merge(tr[x].r, y);
		pushup(x);
		return x;
	} 
	else {
		tr[y].l = merge(x, tr[y].l);
		pushup(y);
		return y;
	}
}

void insert(int key) {
	int x, y;
	split(Root, key - 1, x, y);
	Root = merge(merge(x, create(key)), y);
}

void remove(int key) {
	int x, y, z;
	split(Root, key, x, z);
	split(x, key - 1, x, y);
	y = merge(tr[y].l, tr[y].r);
	Root = merge(merge(x, y), z);
}

int getRank(int key) {
	int x, y, ans;
	split(Root, key - 1, x, y);
	ans = tr[x].size + 1;
	Root = merge(x, y);
	return ans;
}

int kth(int r) {
	int root = Root;
	while (true) {
		if (tr[tr[root].l].size + 1 == r) {
			break;
		}
		else if (tr[tr[root].l].size + 1 > r) {
			root = tr[root].l;
		} 
		else {
			r -= tr[tr[root].l].size + 1;
			root = tr[root].r;
		}
	}
	return tr[root].val;
}

int lower(int key) {
	int x, y, root, ans;
	split(Root, key - 1, x, y);
	root = x;
	while (tr[root].r) root = tr[root].r;
	ans = tr[root].val;
	Root = merge(x, y);
	return ans;
}

int upper(int key) {
	int x, y, root, ans;
	split(Root, key, x, y);
	root = y;
	while (tr[root].l) root = tr[root].l;
	ans = tr[root].val;
	Root = merge(x, y);
	return ans;
}

int main() {
	scanf("%d", &n);
	for (int i = 1; i <= n; ++i) {
		scanf("%d %d", &opt, &x);
		if (opt == 1) insert(x);
		if (opt == 2) remove(x);
		if (opt == 3) printf("%d\n", getRank(x));
		if (opt == 4) printf("%d\n", kth(x));
		if (opt == 5) printf("%d\n", lower(x));
		if (opt == 6) printf("%d\n", upper(x));
	}
	return 0;
}
```

## 可持久化数据结构

### 可持久化权值线段树

* 给定 *n* 个整数构成的序列 *a*，将对于指定的闭区间[l, r]查询其区间内的第 *k* 小值。

```c++
#include<bits/stdc++.h>

using namespace std;
using ll = long long;

const int maxn = 1e5 + 7;

struct HJT {
	int l, r, val;
}tr[maxn << 5];

vector<int> rt(maxn, 0), a(maxn, 0), lsh;
int cnt = 0;

void insert(int pre, int& now, int l, int r, int x) {
	tr[++cnt] = tr[pre];
	now = cnt;
	tr[now].val++;
	if(l == r) return ;
	int mid = l + r >> 1;
	if(x <= mid) insert(tr[pre].l, tr[now].l, l, mid, x);
	else         insert(tr[pre].r, tr[now].r, mid + 1, r, x);
	return ;
}

int query(int L, int R, int l, int r, int k) {
	if(l == r) return l;
	int mid = l + r >> 1;
	int tmp = tr[tr[R].l].val - tr[tr[L].l].val;
	if(k <= tmp) return query(tr[L].l, tr[R].l, l, mid, k);
	else         return query(tr[L].r, tr[R].r, mid + 1, r, k - tmp);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

	int n;
	cin >> n;

	for(int i = 1;i <= n;i++) cin >> a[i];
	for(int i = 1;i <= n;i++) lsh.push_back(a[i]);
	sort(lsh.begin(), lsh.end());
	lsh.erase(unique(lsh.begin(), lsh.end()), lsh.end());
	map<int, int> mp;
	for(int i = 0;i < lsh.size();i++) mp[lsh[i]] = i + 1;
	for(int i = 1;i <= n;i++) insert(rt[i - 1], rt[i], 1, n, mp[a[i]]);
	int m;
	cin >> m;
	for(int i = 1;i <= m;i++) {
		int l, r, k;
		cin >> l >> r >> k;
		l += 1, r += 1;
        //k = r - l + 2 - k;  //第k大
        //k;                  //第k小
		cout << lsh[query(rt[l - 1], rt[r], 1, n, k) - 1] << '\n';
	}

    return 0;
}
```

### 可持久化数组

* 在某个历史版本上修改某一个位置上的值

* 访问某个历史版本上的某一位置的值(**对于操作2，即为生成一个完全一样的版本，不作任何改动**)

```c++
#include<bits/stdc++.h>

using namespace std;
using ll = long long;

const int maxn = 1e6 + 7;

struct node {
	int l, r, val;
}tr[maxn * 20];

vector<int> rt(maxn, 0);
int cnt = 0;

void build(int& p, int l, int r) {
	p = ++cnt;
	if(l == r) {
		cin >> tr[p].val;
		return ;
	}
	int mid = l + r >> 1;
	build(tr[p].l, l, mid);
	build(tr[p].r, mid + 1, r);
	return ;
}

void insert(int pre, int &now, int l, int r, int loc, int x) {
	now = ++cnt;
	tr[now] = tr[pre];
	if(l == r) {
		tr[now].val = x;
		return ;
	}
	int mid = l + r >> 1;
	if(loc <= mid) insert(tr[pre].l, tr[now].l, l, mid, loc, x);
	else insert(tr[pre].r, tr[now].r, mid + 1, r, loc, x);
}

int query(int now, int l, int r, int x) {
	if(l == r) return tr[now].val;
	int mid = l + r >> 1;
	if(x <= mid) return query(tr[now].l, l, mid, x);
	else return query(tr[now].r, mid + 1, r, x);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

	int n, m;
	cin >> n >> m;
	build(rt[0], 1, n);
	for(int i = 1;i <= m;i++) {
		int ver, op, loc;
		cin >> ver >> op >> loc;
		if(op == 1) {
			int x;
			cin >> x;
			insert(rt[ver], rt[i], 1, n, loc, x);
		}
		else {
			cout << query(rt[ver], 1, n, loc) << '\n';
			rt[i] = rt[ver];
		}
	}

    return 0;
}
```

### 可持久化并查集

### 可持久化FHQ-Treap

# Number 数论

## 素数筛

### 欧拉筛

> 对if(i % prime[j] == 0)  break;的解释
>
> 当i % prime[j] == 0时有 i = k * prime[j];   若j++有   i * prime[j + 1] = k * prime[j] * prime[j + 1] 也是prime[j]的因子，导致重复筛

```c++
const int maxn = 1e7 + 8;
vector<int> prime(maxn, 0);
bitset<maxn> vis(0);
void getPrime() {
	for(int i = 2;i < maxn;i++) {
		if(vis[i] == 0) prime[++prime[0]] = i;
		for(int j = 1;j <= prime[0];j++) {
			if(i * prime[j] >= maxn) break;
			vis[i * prime[j]] = 1;
			if(i % prime[j] == 0) break;
		}
	}
}
```

### 埃式筛

```c++
void getPrime(){
    vector<int > vis(maxn, 0); //初始化都是素数
    vis[0] = vis[1] = 1;    //0 和 1不是素数
    for (int i = 2; i <= maxn; i++) {
        if (!vis[i]) {      //如果i是素数，让i的所有倍数都不是素数
            for (int j = i * i; j <= maxn; j += i) { 
                vis[j] = 1;
            }
        }
    }
}
```

### 任意区间素数筛

```c++
const int M=1e6+5,N=(1<<16);
int prime[N+5],is[N+5],tot;
void getPrime(){
    is[1]=1;//mmpaaaaaaaaaaaaaaaa
    for(int i=2;i<=N;++i){
        if(!is[i]) prime[++tot]=i;
        for(int j=1;j<=tot&&(ll)i*prime[j]<=N;++j){
            is[i*prime[j]]=1;
            if(i%prime[j]==0) break;
        }
    }
}
int issp[M];
int main(){
    getPrime();
    int T,cas=0;scanf("%d",&T);
    while(T--){
        m(issp,0);
        int a,b;scanf("%d%d",&a,&b);
        if(a<=N&&b<=N) {
            int ans=0;
            for(int i=a;i<=b;++i) if(!is[i]) ++ans;
            printf("Case %d: %d\n",++cas,ans);
            continue;
        }
        int ans=0;
        if(a<=N) {
            for(int i=a;i<=N;++i) if(!is[i]) ++ans;
            a=N+1;
        }
        for(int i=1;i<=tot;++i) {
            ll l=a/prime[i]*prime[i];//左端点l.
            if(l<a) l+=prime[i];
            if(l==prime[i]) l+=prime[i];
            for(ll j=l;j<=b;j+=prime[i]) issp[j-a]=1;
        }
        for(int j=a;j<=b;++j) if(!issp[j-a]) ++ans;
        printf("Case %d: %d\n",++cas,ans);
    }
}
```

### 大数素数测试

#### Miller_Rabin

```c++
#include <bits/stdc++.h>

using namespace std;
using ll = __int128;

const int S = 8; //随机算法判定次数一般 8～10 就够了
// 计算 ret = (a*b)%c a,b,c < 2^63
__int128 mult_mod(__int128 a, __int128 b, __int128 c) {
    a %= c;
    b %= c;
    __int128 ret = 0;
    __int128 tmp = a;

    while (b) {
        if (b & 1) {
            ret += tmp;

            if (ret > c)
                ret -= c;//直接取模慢很多
        }

        tmp <<= 1;

        if (tmp > c)
            tmp -= c;

        b >>= 1;
    }

    return ret;
}
// 计算 ret = (a^n)%mod
__int128 pow_mod(__int128 a, __int128 n, __int128 mod) {
    __int128 ret = 1;
    __int128 temp = a % mod;

    while (n) {
        if (n & 1)
            ret = mult_mod(ret, temp, mod);

        temp = mult_mod(temp, temp, mod);
        n >>= 1;
    }

    return ret;
}
bool check(__int128 a, __int128 n, __int128 x, __int128 t) {
    __int128 ret = pow_mod(a, x, n);
    __int128 last = ret;

    for (int i = 1; i <= t; i++) {
        ret = mult_mod(ret, ret, n);

        if (ret == 1 && last != 1 && last != n - 1)
            return true;//合数

        last = ret;
    }

    if (ret != 1)
        return true;
    else
        return false;
}
//**************************************************
// Miller_Rabin 算法
// 是素数返回 true,(可能是伪素数)
// 不是素数返回 false
//**************************************************
bool MR(__int128 n) {
    if (n < 2)
        return false;

    if (n == 2)
        return true;

    if ((n & 1) == 0)
        return false;//偶数

    __int128 x = n - 1;
    __int128 t = 0;

    while ((x & 1) == 0) {
        x >>= 1;
        t++;
    }

    srand(time(NULL));

    for (int i = 0; i < S; i++) {
        __int128 a = rand() % (n - 1) + 1;

        if (check(a, n, x, t))
            return false;
    }

    return true;
}
inline __int128 read() {
    __int128 x = 0, f = 1;
    char ch = getchar();

    while (ch < '0' || ch > '9') {
        if (ch == '-')
            f = -1;

        ch = getchar();
    }

    while (ch >= '0' && ch <= '9') {
        x = x * 10 + ch - '0';
        ch = getchar();
    }

    return x * f;
}
int main() {
    __int128 n = read();

    if (MR(n))
        puts("Yes");
    else
        puts("No");

    return 0;
}
```

#### Miller_Rabin(Easy ver)

```c++
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

ll mul(ll a, ll b, ll p) {
	return (__int128)a * b % p;
}

ll qp(ll x, ll y, ll p) {
	ll z = 1;
	for (; y; y >>= 1, x = mul(x, x, p)) {
		if (y & 1) z = mul(z, x, p);
	}
	return z;
}

bool MR(ll x, ll b) {
	for (ll k = x - 1; k; k >>= 1) {
		ll cur = qp(b, k, x);
		if (cur != 1 && cur != x - 1) return 0;
		if (k & 1 || cur == x - 1) return 1;
	}
	return true;
}

bool test(ll x) {
	if (x == 1) return 0;
	static ll p[] = {2, 3, 5, 7, 17, 19, 61};
	for (ll y : p) {
		if (x == y) return 1;
		if (!MR(x, y)) return 0;
	}
	return 1;
}

int main() {
	ll n;
	while (scanf("%lld", &n) != EOF) {
		puts(test(n) ? "Y" : "N");
	}
	return 0;
}
```

## 欧拉函数

​	用处：1：求1-n内与n互质的数量

​				2：求合数的逆元

​				3：欧拉降幂

```c++
int phi(int x) {
	int ans = x;
	for(int i = 2;i * i <= x;i++) {
		if(x % i == 0) {
			ans = ans / i * (i - 1);
			while(x % i == 0) x /= i;
		}
	}
	if(x > 1) ans = ans / x * (x - 1);
	return ans;
}
```

### 线性筛欧拉函数

``` c++
const int maxn = 100005;
vector<int > prime(maxn, 0), vis(maxn, 0), phi(maxn, 0);
void getPrime() {
    phi[1] = 1;
	for(int i = 2;i < maxn;i++) {
		if(vis[i] == 0) prime[++prime[0]] = i, phi[i] = i - 1;
		for(int j = 1;j <= prime[0];j++) {
			if(i * prime[j] >= maxn) break;
			vis[i * prime[j]] = 1;
			if(i % prime[j] != 0) phi[i * prime[j]] = phi[i] * (prime[j] - 1);
            //对于任意的a，b若gcd(a, b) == 1 -> phi(a * b) = phi(a) * phi(b)
			else {
                phi[i * prime[j]] = phi[i] * prime[j];
                break;
            }
		}
	}
}
```

## Mobius

### Mobius反演

1. **Mobius函数反演的第一种形式,其中f(n)为n的因数和；**

$$
f(n)=\sum_{d|n}^{}g(d){\iff}g(n)=\sum_{d|n}\mu(\frac{n}{d})·f(d)
$$

2. **Mobius函数反演的第二种形式,其中f(n)为gcd为n的倍数的关系求和；**

$$
f(n)=\sum_{n|m}^{}g(m){\iff}g(n)=\sum_{n|m}\mu(\frac{m}{n})·f(n)
$$

### 线性筛Mobius函数

$$
\mu(i)=\left\{
\begin{matrix} 
(-1)^{k} && p_1p_2···p_k&(p_i为素数)\\ 
0 && i有平方因子\\
1 && i = 1
\end{matrix}
\right.
$$

```c++
constexpr int maxn = 1e5 + 7;

int prime[maxn], vis[maxn], mobius[maxn];

void getMobius() {
	mobius[1] = 1;
	for(int i = 2;i < maxn;i++) {
		if(vis[i] == 0) {
			prime[++prime[0]] = i;
			mobius[i] = -1;
		}
		for(int j = 1;j <= prime[0];j++) {
			if(i * prime[j] >= maxn) break;
			vis[i * prime[j]] = 1;
			if(i % prime[j] == 0) {
				mobius[i * prime[j]] = 0;
				break;
			}
			else {
				mobius[i * prime[j]] = -mobius[i];
			}
		}
	}
}
```

## 线性基

#### **一、概念**

在线性代数中，对于向量组 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_1%2C%5Calpha_2%2C%5Cdots%2C%5Calpha_n) ，我们把其张成空间的一组**线性无关**的基成为该向量组的线性基。

二进制集合 ![[公式]](https://www.zhihu.com/equation?tex=S%3D%5Cleft+%5C%7B+x_1%2Cx_2%2C%5Cdots%2Cx_n%5Cright+%5C%7D) ，得到另一个二进制集合 ![[公式]](https://www.zhihu.com/equation?tex=S%27%3D%5Cleft+%5C%7B++y_1%2Cy_2%2C%5Cdots+y_n%5Cright+%5C%7D) ，保证在 ![[公式]](https://www.zhihu.com/equation?tex=S) 中任取子集 ![[公式]](https://www.zhihu.com/equation?tex=A) ，都能在 ![[公式]](https://www.zhihu.com/equation?tex=S%27) 中找到对应的子集 ![[公式]](https://www.zhihu.com/equation?tex=A%27) ，使得 ![[公式]](https://www.zhihu.com/equation?tex=A) 与 ![[公式]](https://www.zhihu.com/equation?tex=A%27) 的异或和相等；同时 ![[公式]](https://www.zhihu.com/equation?tex=S%27) 中任意一个元素都不能被 ![[公式]](https://www.zhihu.com/equation?tex=S%27) 中其他元素的组合异或出来。我们把 ![[公式]](https://www.zhihu.com/equation?tex=S%27) 称为 ![[公式]](https://www.zhihu.com/equation?tex=S) 的**线性基**，利用它可以方便的求出原集合的**k大异或和**。 *（来自我喜欢的知乎博主Pecco）*

#### **二、理解**

对概念有一定初级的了解之后，我想说一下我的理解： 1）**关于线性基**：线性基就好比一堆向量（数），可以用若干个线性基里面的来表示一个集合里面的所有数，他们互相**线性无关**。 2）**关于线性无关**：就是线性代数里面的，线性基里面的每个数都有贡献，原集合里面的数只能在线性基里面找一组**唯一**的子集来通过异或和来表示。 3）**总的来看**，就好比我们高中学过的多维向量，我们可以用一组**基底**来表示原来所有的向量，而且唯一。

#### **三、构造**

线性基的构造一般由两种方法，一种是用数组来，一种是用vector的简单方法，我看大部分的题目还是用的前者，所以建议还是用前者。

##### **数组**

##### **理解p数组**

我们这里是有一个p数组，对于一般的1e18数据，我们开最多循环62位，所以p数组开到65、100就行了。 **p[i]代表的是最高有效位为第i位的线性基的向量**，怎么理解呢，比如我们插入`x=01001101`，那么我们会判断`x&(1<<i)`时候(此时`i=6`)时候有效，将`p[6]=01001101`，因为最高有效位为第六位（这里包括第0位），大致懂了吧，**也就是说，我们线性基最多有62位（63个，包括第0位），我这个时候其中有一个是01001101**

##### **理解动态和构造**

为什么说动态呢，因为我们线性基是会变的，举一个简单的例子，如果这时候原集合为 ![[公式]](https://www.zhihu.com/equation?tex=S%3D%5Cleft+%5C%7B01001101%2C01101001%5Cright+%5C%7D) ，那么我们此时将`x=01101001`插入进来，再线性基的判断的时候（从高往低），我们会将判断到最高位到第六位的时候，此时`p[6]`已经有值了，所以我们**将x进行异或操作**，通过`x^=p[i]`，将x更新为`x=00100100`，此时我们继续循环`x`的最高有效位，到第五位的时候，我们会判断到`p[5]`此时没有数，所以我们将`x`赋值给`p[5]`，此时`p[5]=00100100`

我们再来看线性基，里面已经有两个**基**了，分别是`p[6]=01001101`和`p[5]=01001101`，线性基里面的数时能够通过若干个来异或和来表示原集合里面的所有数，比如原集合的第一个数`a[1]=01001101`，我们会发现就是`p[6]`可以表示，那第二个数`a[2]=01101001`呢，通过构造我们可以发现这个`p[5]`是由于`p[5]=a[2]^a[6]`而来的，所以式子两边同时异或`a[6]`就可以得到`a[2] = p[5] ^ p[6]`，我们用到了以下的两个异或性质

1）`a ^ b =c ==> a = c ^ b`

2）`(a ^ b) ^ b = a`

所以我们的线性基是动态的，一直是变化的。

```cpp
bool insert(vector<int> &p, int x) {//Aim : We need to construction
    for(int i = 30; i >= 0; i--) {
        //if(x == Aim) return ;     Succseeful
        if(x >> i) {
            if(p[i]) {
                x ^= p[i];
			}
			else {
				p[i] = x;
				return 1;
			}
		}
	}
	return 0;
}
```

## 数论分块

面对这样的一类问题
$$
\sum_{i=1}^{n}\lfloor{\frac{n}{i}}\rfloor
$$
不难发现$\lfloor{\frac{n}{i}}\rfloor$ 的值在一段连续区间上的值是相等的，所以我们可以利用数论分块进行解决

```cpp
for(int l = 1, r = 0; l <= n; l = r + 1) {
    r = n / (n / l);
    //[l, r] : n / i is Same
}
```



## 逆元

### EXGCD求a在b意义下的逆元

$$
condition:[gcd(a, b)=1]
$$

```c++
int exgcd(int a, int b, int &x, int &y) {
	if(b == 0) {
		x = 1;
		y = 0;
		return a;
	}
	else {
		int gcd = exgcd(b, a % b, x, y);
		int t = x;
		x = y;
		y = t - a / b * y;
		return gcd;
	}
}
int getInv(int a, int b){
	int x, y;
	exgcd(a, b, x, y);
	return (x + b) % b;
}
```

### 费马小定理求a在mod p意义下的逆元(p是素数)

$$
a * a^{p - 2} = 1{(modp)}
$$

```c++
int qpow(int a, int b, int mod) {
	int ans = 1;
	while(b) {
		if(b & 1) ans = ans * a % mod;
		b >>= 1, a = a * a % mod;
	}
	return ans;
}
int getInv(int x, int mod) {
	return qpow(x, mod - 2, mod);
}

```

### 费马小定理求a在mod p意义下的逆元(p不是素数)

$$
a*a^{phi(p) - 1} = 1(modp)
$$

```c++
int qpow(int a, int b, int mod) {
	int ans = 1;
	while(b) {
		if(b & 1) ans = ans * a % mod;
		b >>= 1, a = a * a % mod;
	}
	return ans;
}
int phi(int x) {
	int ans = x;
	for(int i = 2;i * i <= x;i++) {
		if(x % i == 0) {
			ans = ans * (i - 1) / i;
			while(x % i == 0) x /= i;
		}
	}
	if(x > 1) ans = ans * (x - 1) / x;
	return ans;
}
int getInv(int x, int mod) {
	return qpow(x, phi(mod) - 1, mod);
}

```

### O(n)求逆元

``` c++
vector<ll> inv(maxn, 0);
inv[1] = 1;
for(int i = 2;i < maxn;i++) inv[i] = 1ll * (p - p / i) * inv[p % i] % p;
```

### O(n)求阶乘逆元

```cpp
int inv( int b, int p ) {
    int a, k;
    exPower( b, p, a, k );
    if( a < 0 ) a += p;
    return a;
}
void init( int n ) {
    Fact[ 0 ] = 1;
    for( int i = 1; i <= n; ++i ) Fact[ i ] = Fact[ i - 1 ] * i % Mod;
    INV[ n ] = inv( Fact[ n ], Mod );
    for( int i = n - 1; i >= 0; --i ) INV[ i ] = INV[ i + 1 ] * ( i + 1 ) % Mod;
    return;
}
```

### O(n + log(mod))求任意n个数的逆元

```c++
const ll mod  = 1e9 + 7;
const ll p    = 998244353;
const ll maxn = 1e5 + 7;

vector<ll> a(maxn, 0), pre(maxn, 1), pre_inv(maxn, 0), inv(maxn, 0);
//		   数组元素     前缀积		    前缀积的逆元	       每个数的逆元

ll qpow(ll a, ll b) {
	ll ret = 1;
	while(b) {
		if(b & 1) ret = ret * a % mod;
		b >>= 1, a = a * a % mod;
	}
	return ret;
}

void solve(int n) {//数组长度参数
	for(int i = 1;i <= n;i++) cin >> a[i];
	for(int i = 1;i <= n;i++) pre[i] = pre[i - 1] * a[i] % mod;
	pre_inv[n] = qpow(pre[n], mod - 2);
	for(int i = n;i >= 1;i--) pre_inv[i - 1] = pre_inv[i] * a[i] % mod;
	for(int i = 1;i <= n;i++) inv[i] = pre_inv[i] * pre[i - 1] % mod;
}
```

## 组合数

![img](https://img-blog.csdnimg.cn/img_convert/bb29d02a740d339f7cceb76f73fbcee9.png)

### n大m小组合数-O(m)

```c++
const int mod = 1e9 + 7;

struct Binom {
	int n;
	vector<ll> inv;
	Binom(int n_ = 0) : n(n_), inv(n) {
		inv[1] = 1;
		for(int i = 2; i < n; i++) inv[i] = ((-(mod / i) * inv[mod % i]) % mod + mod) % mod;
	}
	ll C(ll n, ll m) { // m in a small range
		ll ret = 1;
		for(int i = 0; i < m; i++)
			ret *= (n - i) % mod, ret %= mod, ret *= inv[i + 1], ret %= mod;
		return ret;
	}
};
```

### 正常版本-O(logx)

$$
C_n^m = \frac{n!}{m!*(n - m)!}
$$

```c++
struct Combination {
	const int mod = 1e9 + 7;
	int n;
	vector<ll> fac, inv;

	Combination(int n_ = 0) : n(n_) {
		fac.resize(n << 1, 1);
		inv.resize(n << 1, 1);
		for(ll i = 1;i <= n_;i++) fac[i] = fac[i - 1] * i % mod;
		inv[n] = qpow(fac[n], mod - 2);  
		for(ll i = n - 1;i >= 1;i--) inv[i] = (inv[i + 1] * (i + 1)) % mod;
	}

	ll qpow(ll a, ll b) {
		ll ret = 1;
		while(b) {
			if(b & 1) ret = ret * a % mod;
			b >>= 1, a = a * a % mod;
		}
		return ret;
	}

	ll c(ll n, ll m) {
		if(!m) return 1;
		if(m > n) return 0;
		return fac[n] * inv[m] % mod * inv[n - m] % mod;
	}
};
```

### Mint版本

```c++
constexpr int mod = 1000000007;
constexpr int maxn = 1e5 + 6;
// assume -mod <= x < 2mod
int norm(int x) {if(x < 0) x += mod; if (x >= mod) x -= mod; return x; }
template<class T>T power(T a, ll b){T res = 1;for (; b; b /= 2, a *= a)if (b & 1)res *= a;return res;}
struct Mint {
    int x;Mint(int x = 0) : x(norm(x)){}
    int val() const {return x;}
    Mint operator-() const {return Mint(norm(mod - x));}
    Mint inv() const {assert(x != 0);return power(*this, mod - 2);}
    Mint &operator*=(const Mint &rhs) { x = ll(x) * rhs.x % mod; return *this;}
    Mint &operator+=(const Mint &rhs) { x = norm(x + rhs.x); return *this;}
    Mint &operator-=(const Mint &rhs) { x = norm(x - rhs.x); return *this;}
    Mint &operator/=(const Mint &rhs) {return *this *= rhs.inv();}
    friend Mint operator*(const Mint &lhs, const Mint &rhs) {Mint res = lhs; res *= rhs; return res;}
    friend Mint operator+(const Mint &lhs, const Mint &rhs) {Mint res = lhs; res += rhs; return res;}
    friend Mint operator-(const Mint &lhs, const Mint &rhs) {Mint res = lhs; res -= rhs; return res;}
    friend Mint operator/(const Mint &lhs, const Mint &rhs) {Mint res = lhs; res /= rhs; return res;}
};

vector<Mint> fac(maxn, 1);

void init() {
	for(ll i = 1;i < maxn;i++) fac[i] = fac[i - 1] * i;
}

Mint c(ll n, ll m) {
	return fac[n] / fac[m] / fac[n - m];
}
```

### Lucas

```c++
const int maxn = 2e5 + 7;
vector<ll> fac(maxn, 1);

void init(ll p) {//if module is different
	for(ll i = 1;i <= 200000;i++) fac[i] = fac[i - 1] * i % p;
}

ll qpow(ll a, ll b, ll p) {
	ll ret = 1;
	while(b) {
		if(b & 1) ret = ret * a % p;
		b >>= 1;
		a = a * a % p;
	}
	return ret;
}

ll getInv(ll x, ll p) {
	if(x == 1) return 1;
	return qpow(x, p - 2, p);
}

ll c(ll n, ll m, ll p) {
	if(!m) return 1;
	if(m > n) return 0;
	return fac[n] * getInv(fac[m], p) % p * getInv(fac[n - m], p) % p;
}

ll lucas(ll n, ll m, ll p) {
	if(n == 0) return 1;
	return lucas(n / p, m / p, p) * c(n % p, m % p, p) % p;
}
```

## 欧几里得

### GCD

### EXGCD

```c++
int exgcd(int a, int b, int &x, int &y) {
	if(b == 0) {
		x = 1;
		y = 0;
		return a;
	}
	else {
		int gcd = exgcd(b, a % b, x, y);
		int t = x;
		x = y;
		y = t - a / b * y;
		return gcd;
	}
}
```

### CRT

``` c++
using ll = long long;

ll qpow(ll a, ll b, ll mod) {
	ll ret = 1;
	while(b) {
		if(b & 1) ret = ret * a % mod;
		b >>= 1;
		a = a * a % mod;
	}
	return ret;
}

ll getInv(ll x, ll mod) {
	return qpow(x, mod - 2, mod);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

	ll n, mul = 1, sum = 0;
	cin >> n;
	vector<ll> a(n + 1, 0), mu(n + 1, 0);
	for(int i = 1;i <= n;i++) cin >> a[i] >> mu[i];
	for(int i = 1;i <= n;i++) mul *= a[i];
	for(int i = 1;i <= n;i++) {
		ll lcm = mul / a[i];
		ll val = getInv(lcm, a[i]) * mu[i] * lcm;
		sum += val;
	}
	cout << sum % mul << '\n';

    return 0;
}
```

## 矩阵

* 加速递推，$O(k^{3}log_{2}n)$
* 线段树无脑合并

$$
f_{i} = f_{i - 1} + f_{i - 2}:
\begin{bmatrix}
{1}&{1}\\
{1}&{0}\\
\end{bmatrix}
\cdot
\begin{bmatrix}
{f_{n-1}}\\
{f_{n-2}}\\
\end{bmatrix}
=
\begin{bmatrix}
{f_{n}}\\
{f_{n-1}}\\
\end{bmatrix}
$$

$$
f_{i} = a*f_{i - 1} + i: 
\begin{bmatrix}
{a}&{1}&{0}\\
{0}&{1}&{1}\\
{0}&{0}&{1}\\
\end{bmatrix}
\cdot
\begin{bmatrix}
{f_{i-1}}\\
{{i}}\\
{{1}}\\
\end{bmatrix}
=
\begin{bmatrix}
{f_{i}}\\
{{i+1}}\\
{{1}}\\
\end{bmatrix}
$$

### Matrix Class

```c++
struct Matrix { 
	int n, m;
	vector<vector<ll> > a;//If you get TLE, try to instead of vector in array...

	Matrix(int n_ = 0, int m_ = 0) : n(n_), m(m_), a(n + 1, vector<ll>(m + 1, 0)) {}

	Matrix operator + (const Matrix& b) {
		Matrix ret(n, m);
		for(int i = 1;i <= n;i++) 
			for(int j = 1;j <= m;j++) 
				ret.a[i][j] = (a[i][j] + b.a[i][j]) % mod;
		return ret;
	}

	Matrix operator - (const Matrix& b) {
		Matrix ret(n, m);
		for(int i = 1;i <= n;i++) 
			for(int j = 1;j <= m;j++) 
				ret.a[i][j] = (a[i][j] - b.a[i][j] + mod) % mod;
		return ret;
	}

	Matrix operator * (const Matrix& b) {
		Matrix ret(n, b.m);
		for(int i = 1;i <= n;i++) 
			for(int j = 1;j <= b.m;j++) 
				for(int k = 1;k <= m;k++) 
					ret.a[i][j] = (ret.a[i][j] + a[i][k] * b.a[k][j]) % mod;
		return ret;
	}	
};
```

### 矩阵快速幂

```c++
constexpr int mod = 1e9 + 7;

struct Matrix { 
	int n, m;
	vector<vector<ll> > a;//If you get TLE, try to instead of vector in array...

	Matrix(int n_ = 0, int m_ = 0) : n(n_), m(m_), a(n + 1, vector<ll>(m + 1, 0)) {}

	Matrix unitMatrix() {		//to E
		Matrix E(n, m);
		for(int i = 1;i <= n;i++) E.a[i][i] = 1;
		return E;
	}
	
	Matrix f1_() {		//fib(n) = f(n - 1) + f(n - 2),  f[n] = qpow(A, n - 2);
		Matrix ret(n, m);
		ret.a[1][1] = 1, ret.a[1][2] = 1;
		ret.a[2][1] = 1, ret.a[2][2] = 0;
		return ret;
	}

	Matrix f2_() {		//tri(n) = f(n - 1) + f(n - 2) + f(n - 3);
		Matrix ret(n, m);
		ret.a[1][1] = 1, ret.a[1][2] = 1, ret.a[1][3] = 1;
		ret.a[2][1] = 1, ret.a[2][2] = 0, ret.a[2][3] = 0;
		ret.a[3][1] = 0, ret.a[3][2] = 1, ret.a[3][3] = 0;
		return ret;
	}

	Matrix operator * (const Matrix& b) {
		Matrix ret(n, b.m);
		for(int i = 1;i <= n;i++) 
			for(int j = 1;j <= b.m;j++) 
				for(int k = 1;k <= m;k++) 
					ret.a[i][j] = (ret.a[i][j] + a[i][k] * b.a[k][j]) % mod;
		return ret;
	}	

	Matrix qpow(Matrix a, ll b) {
		Matrix ans = unitMatrix();
		while(b) {
			if(b & 1) ans = ans * a;
			b >>= 1,  a = a * a;
		}
		return ans;
	}
	void show() {
		for(int i = 1;i <= n;i++) 
			for(int j = 1;j <= m;j++) 
				cout << a[i][j] << " \n"[j == m];
		return ;
	}
};
```

## 生成函数

>  **生成函数（Generating Function）**，又称母函数，是一种形式幂级数，其每一项的系数可以提供关于这个序列的信息。
>
> 生成函数有许多不同的种类，但大多可以表示为单一的形式

$$
F(x)=\sum_{n}a_{n}k_{n}(x)
$$

> 其中$k_{n}(x)$被称为核函数，不同的核函数会导出不同的生成函数，拥有不同的性质。举个例子：

1. 普通生成函数：$k_{n}(x)=x^{n}$。
2. 指数生成函数：$k_{n}(x)=\frac{x^n}{n!}$。

> 对于生成函数$F(x)$，我们用$[k_n(x)]F(x)$来表示它的第n项核函数对应的系数，也就是$a_{n}$。

### 普通生成函数

#### 定义

> 序列$a$的普通生成函数（Ordinary Generating Function，OGF）定义为形式幂级数：

$$
F(x)=\sum_{n}a_{n}k_{n}(x)
$$

> $a$既可以是有穷序列，也可以是无穷序列。常见的例子（假设$s$以$0$为起点）:
>
> 1. 序列$a=<1,2,3>$的普通生成函数是$F(x)=1+2x+3x^2$。
> 2. 序列$a=<1,1,1....>$的普通生成函数是$F(x)=\sum_{n\geq0}x^n$。
> 3. 序列$a=<1,2,4,8...>的普通生成函数是$$F(x)=\sum_{n\geq0}2^nx^n$。
> 4. 序列$a=<1,3,5,7...>$的普通生成函数是$F(x)=\sum_{n\geq0}(2n+1)x^n$。

#### 封闭形式

> 在运用生成函数的过程中，我们不会一直使用形式幂级数的形式，而会适时地转化为封闭形式以更好地化简。
>
> 例如：序列$a=<1,1,1....>$的普通生成函数是$F(x)=\sum_{n\geq0}x^n$，我们可以发现
> $$
> x·F(x) + 1=F(x)
> $$
> 那么解这个方程得到
> $$
> F(x)=\frac{1}{x-1}
> $$
> 这就是$F(x)=\sum_{n\geq0}x^n$的封闭形式。

$$
\frac{1}{(1-n)^k}=\sum_{i}^{inf}\binom{i+k-1}{k - 1}x^i
$$



![image-20220415203758452](C:\Users\JUNKAIZHANG\AppData\Roaming\Typora\typora-user-images\image-20220415203758452.png)

![image-20220415203845408](C:\Users\JUNKAIZHANG\AppData\Roaming\Typora\typora-user-images\image-20220415203845408.png)

## 卷积&多项式

### FFT

``` c++
const int MAXN = 1e6 + 10;
const double Pi = acos(-1.0);
struct complex{
	double x, y;
	complex (double xx = 0, double yy = 0) {x = xx, y = yy;}
}a[MAXN], b[MAXN];
complex operator + (complex a, complex b) { return complex(a.x + b.x , a.y + b.y);}
complex operator - (complex a, complex b) { return complex(a.x - b.x , a.y - b.y);}
complex operator * (complex a, complex b) { return complex(a.x * b.x - a.y * b.y , a.x * b.y + a.y * b.x);} 
int l, r[MAXN], ans[MAXN];
int limit = 1;
void fft(complex *A, int type) {
	for(int i = 0; i < limit; i++) if (i < r[i]) swap(A[i], A[r[i]]); 
	for (int mid = 1; mid < limit; mid <<= 1) { 
		complex Wn( cos(Pi / mid) , type * sin(Pi / mid) ); 
		for (int R = mid << 1, j = 0; j < limit; j += R) { 
			complex w(1, 0);
			for (int k = 0; k < mid; k++, w = w * Wn) { 
				complex x = A[j + k], y = w * A[j + mid + k]; 
				A[j + k] = x + y;
				A[j + mid + k] = x - y;
			}
		}
	}
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	string aa, bb;
	cin >> aa >> bb;
	int n = aa.size(), m = bb.size();
	for (int i = 0; i < n; i++) a[i].x = aa[i] ^ 0x30;
	for (int i = 0; i < m; i++) b[i].x = bb[i] ^ 0x30;
	while (limit <= n + m) limit <<= 1, l++;
	for (int i = 0; i < limit; i++) r[i] = ( r[i >> 1] >> 1 ) | ( (i & 1) << (l - 1));
	fft(a, 1);
	fft(b, 1);
	for (int i = 0; i <= limit; i++) a[i] = a[i] * b[i];
	fft(a, -1);
	for (int i = 0; i <= n + m; i++) ans[i] = (int)(a[i].x / limit + 0.5);
	for(int i = n + m;i >= 1;i--) ans[i - 1] += ans[i] / 10, ans[i] %= 10;
	for(int i = 0;i <= n + m - 2;i++) cout << ans[i]; cout << '\n';
}
```

### NTT

#### 原根表

有质数 $p=k⋅2^r+1$, 原根为 $g$

| p                  | r    | k    | g    |
| ------------------ | ---- | ---- | ---- |
| 81788929           | 21   | 39   | 7    |
| 104857601          | 21   | 50   | 3    |
| 104857601          | 22   | 25   | 3    |
| 113246209          | 21   | 54   | 7    |
| 113246209          | 22   | 27   | 7    |
| 132120577          | 21   | 63   | 5    |
| 136314881          | 21   | 65   | 3    |
| 138412033          | 21   | 66   | 5    |
| 155189249          | 21   | 74   | 6    |
| 167772161          | 23   | 20   | 3    |
| 998244353          | 23   | 119  | 3    |
| 1004535809         | 21   | 479  | 3    |
| 2717908993         | 22   | 648  | 5    |
| 20501757953        | 25   | 611  | 2    |
| 31525197391593473  |      |      | 3    |
| 180143985094819841 |      |      | 6    |

#### 光速乘

```cpp
ll mul(ll a, ll b) {
    return (__int128)a * (__int128)b % p;
 
    a %= p;
    b %= p;
    return ((a * b - p * (ll)((long double)a / p * b + 0.5)) % p + p) % p;
}
```

#### 多项式快速幂

$$
F(x)=(G(x))^k\newline
ln(F(x))=k{·}ln(G(x))\newline
F(x)=e^{k·ln(G(x))}
$$

#### NTT(Standard)

``` c++
constexpr int P = 998244353;
vector<int> rev, roots{0, 1};

int power(int a, int b) {
	int res = 1;
	for(; b; b >>= 1, a = 1ll * a * a % P) if(b & 1) res = 1ll * res * a % P;
	return res;
}
void dft(vector<int> &a) {
	int n = a.size();
	if(int(rev.size()) != n) {
		int k = __builtin_ctz(n) - 1;
		rev.resize(n);
		for (int i = 0; i < n; ++i) rev[i] = rev[i >> 1] >> 1 | (i & 1) << k;
	}
	for(int i = 0; i < n; ++i) if(rev[i] < i) swap(a[i], a[rev[i]]);
	if(int(roots.size()) < n) {
		int k = __builtin_ctz(roots.size());
		roots.resize(n);
		while((1 << k) < n) {
			int e = power(3, (P - 1) >> (k + 1));
			for(int i = 1 << (k - 1); i < (1 << k); ++i) {
				roots[2 * i] = roots[i];
				roots[2 * i + 1] = 1ll * roots[i] * e % P;
			}
			++k;
		}
	}
	for(int k = 1; k < n; k *= 2) {
		for(int i = 0; i < n; i += 2 * k) {
			for(int j = 0; j < k; ++j) {
				int u = a[i + j];
				int v = 1ll * a[i + j + k] * roots[k + j] % P;
				int x = u + v;
				if(x >= P) x -= P;
				a[i + j] = x;
				x = u - v;
				if(x < 0) x += P;
				a[i + j + k] = x;
			}
		}
	}
}
void idft(vector<int> &a) {
	int n = a.size();
	reverse(a.begin() + 1, a.end());
	dft(a);
	int inv = power(n, P - 2);
	for(int i = 0; i < n; ++i) a[i] = 1ll * a[i] * inv % P;
}
struct Poly {
	vector<int> a;
	Poly(){}
	Poly(int a0) { if (a0) a = {a0}; }
	Poly(const vector<int> &a1) : a(a1) {
		while(!a.empty() && !a.back()) a.pop_back();
	}
	int size() const { return a.size(); }
	int operator[](int idx) const { if (idx < 0 || idx >= size()) return 0; return a[idx]; }
	Poly mulxk(int k) const {
		auto b = a;
		b.insert(b.begin(), k, 0);
		return Poly(b);
	}
    Poly mulInt(int x) const {
		auto b = a;
		for(int i = 0;i < (int)(b.size());i++) 
			b[i] = 1ll * b[i] * x % P;
		return Poly(b);
	}
	Poly modxk(int k) const {
		k = min(k, size());
		return Poly(vector<int>(a.begin(), a.begin() + k));
	}
	Poly divxk(int k) const {
		if (size() <= k) return Poly();
		return Poly(vector<int>(a.begin() + k, a.end()));
	}
	friend Poly operator+(const Poly a, const Poly &b) {
		vector<int> res(max(a.size(), b.size()));
		for (int i = 0; i < int(res.size()); ++i) {
			res[i] = a[i] + b[i];
			if (res[i] >= P) res[i] -= P;
		}
		return Poly(res);
	}
	friend Poly operator-(const Poly a, const Poly &b) {
		vector<int> res(max(a.size(), b.size()));
		for (int i = 0; i < int(res.size()); ++i) {
			res[i] = a[i] - b[i];
			if (res[i] < 0) res[i] += P;
		}
		return Poly(res);
	}
	friend Poly operator*(Poly a, Poly b) {
		int sz = 1, tot = a.size() + b.size() - 1;
		while (sz < tot) sz *= 2;
		a.a.resize(sz);
		b.a.resize(sz);
		dft(a.a);
		dft(b.a);
		for (int i = 0; i < sz; ++i) a.a[i] = 1ll * a[i] * b[i] % P;
		idft(a.a);
		return Poly(a.a);
	}
	Poly &operator+=(Poly b) { return (*this) = (*this) + b; }
	Poly &operator-=(Poly b) { return (*this) = (*this) - b; }
	Poly &operator*=(Poly b) { return (*this) = (*this) * b; }
	Poly deriv() const {
		if (a.empty()) return Poly();
		vector<int> res(size() - 1);
		for (int i = 0; i < size() - 1; ++i) res[i] = 1ll * (i + 1) * a[i + 1] % P;
		return Poly(res);
	}
	Poly integr() const {
		if(a.empty()) return Poly();
		vector<int> res(size() + 1);
		for (int i = 0; i < size(); ++i) res[i + 1] = 1ll * a[i] * power(i + 1, P - 2) % P;
		return Poly(res);
	}
	Poly inv(int m) const {
		Poly x(power(a[0], P - 2));
		int k = 1;
		while(k < m) {
			k *= 2;
			x = (x * (2 - modxk(k) * x)).modxk(k);
		}
		return x.modxk(m);
	}
	Poly log(int m) const { return (deriv() * inv(m)).integr().modxk(m); }
	Poly exp(int m) const {
		Poly x(1);
		int k = 1;
		while (k < m) {
			k *= 2;
			x = (x * (1 - x.log(k) + modxk(k))).modxk(k);
		}
		return x.modxk(m);
	}
	Poly sqrt(int m) const {
		Poly x(1);
		int k = 1;
		while(k < m) {
			k *= 2;
			x = (x + (modxk(k) * x.inv(k)).modxk(k)) * ((P + 1) / 2);
		}
		return x.modxk(m);
	}
	Poly mulT(Poly b) const {
		if (b.size() == 0) return Poly();
		int n = b.size();
		reverse(b.a.begin(), b.a.end());
		return ((*this) * b).divxk(n - 1);
	}
	vector<int> eval(vector<int> x) const {
		if (size() == 0) return vector<int>(x.size(), 0);
		const int n = max(int(x.size()), size());
		vector<Poly> q(4 * n);
		vector<int> ans(x.size());
		x.resize(n);
		function<void(int, int, int)> build = [&](int p, int l, int r) {
			if (r - l == 1) {
				q[p] = vector<int>{1, (P - x[l]) % P};
			}
			else {
				int m = (l + r) / 2;
				build(2 * p, l, m);
				build(2 * p + 1, m, r);
				q[p] = q[2 * p] * q[2 * p + 1];
			}
		};
		build(1, 0, n);
		function<void(int, int, int, const Poly &)> work = [&](int p, int l, int r, const Poly &num) {
			if (r - l == 1) {
				if (l < int(ans.size()))
					ans[l] = num[0];
			} 
			else {
				int m = (l + r) / 2;
				work(2 * p, l, m, num.mulT(q[2 * p + 1]).modxk(m - l));
				work(2 * p + 1, m, r, num.mulT(q[2 * p]).modxk(r - m));
			}
		};
		work(1, 0, n, mulT(q[1].inv(n)));
		return ans;
	}
};
```

##### 差的卷积

```c++
int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	int n;
	cin >> n;

	vector<int> f;
	for (int i = 0; i < n; i++) {
		int a;
		cin >> a;
		if (a >= int(f.size())) {
			f.resize(a + 1);
		}
		f[a] = 1;
	}

	int mx = f.size() - 1;

	auto g = f;
	reverse(g.begin(), g.end());  //位置取反  //差的卷积
	auto h = Poly(f) * Poly(g);   //h 数组从mx开始为差值为0，后边以此类推


	return 0;
}
```

#### NTT(Simplify)

```c++
constexpr int P = 998244353;
vector<int> rev, roots{0, 1};

int power(int a, int b) {
	int res = 1;
	for(; b; b >>= 1, a = 1ll * a * a % P) if(b & 1) res = 1ll * res * a % P;
	return res;
}
void dft(vector<int> &a) {
	int n = a.size();
	if(int(rev.size()) != n) {
		int k = __builtin_ctz(n) - 1;
		rev.resize(n);
		for (int i = 0; i < n; ++i) rev[i] = rev[i >> 1] >> 1 | (i & 1) << k;
	}
	for(int i = 0; i < n; ++i) if(rev[i] < i) swap(a[i], a[rev[i]]);
	if(int(roots.size()) < n) {
		int k = __builtin_ctz(roots.size());
		roots.resize(n);
		while((1 << k) < n) {
			int e = power(3, (P - 1) >> (k + 1));
			for(int i = 1 << (k - 1); i < (1 << k); ++i) {
				roots[2 * i] = roots[i];
				roots[2 * i + 1] = 1ll * roots[i] * e % P;
			}
			++k;
		}
	}
	for(int k = 1; k < n; k *= 2) {
		for(int i = 0; i < n; i += 2 * k) {
			for(int j = 0; j < k; ++j) {
				int u = a[i + j];
				int v = 1ll * a[i + j + k] * roots[k + j] % P;
				int x = u + v;
				if(x >= P) x -= P;
				a[i + j] = x;
				x = u - v;
				if(x < 0) x += P;
				a[i + j + k] = x;
			}
		}
	}
}
void idft(vector<int> &a) {
	int n = a.size();
	reverse(a.begin() + 1, a.end());
	dft(a);
	int inv = power(n, P - 2);
	for(int i = 0; i < n; ++i) a[i] = 1ll * a[i] * inv % P;
}
struct Poly {
	vector<int> a;
	Poly(){}
	Poly(int a0) { if (a0) a = {a0}; }
	Poly(const vector<int> &a1) : a(a1) {
		while(!a.empty() && !a.back()) a.pop_back();
	}
	int size() const { return a.size(); }
	int operator[](int idx) const { 
		if (idx < 0 || idx >= size()) 
			return 0; 
		return a[idx]; 
	}
	friend Poly operator*(Poly a, Poly b) {
		int sz = 1, tot = a.size() + b.size() - 1;
		while (sz < tot) sz *= 2;
		a.a.resize(sz);
		b.a.resize(sz);
		dft(a.a);
		dft(b.a);
		for (int i = 0; i < sz; ++i) a.a[i] = 1ll * a[i] * b[i] % P;
		idft(a.a);
		return Poly(a.a);
	}
	Poly &operator*=(Poly b) { return (*this) = (*this) * b; }
};
```

### FWT

#### OR

```C++
void fwt_or(vector<ll> &a, int len) {
    for (int mid = 2; mid <= len; mid <<= 1) {
        for (int i = 0; i < len; i += mid) {
            for (int j = i; j < i + (mid >> 1); j++) {
                a[j + (mid >> 1)] += a[j];
			}
		}
	}
	return ;
}
void ifwt_or(vector<ll> &a, int len) {
    for (int mid = 2; mid <= len; mid <<= 1) {
        for (int i = 0; i < len; i += mid) {
            for (int j = i; j < i + (mid >> 1); j++) {
                a[j + (mid >> 1)] -= a[j];
			}
		}
	}
	return ;
}
```

#### AND

```C++
void fwt_and(vector<ll> &a, int len) {
    for (int mid = 2; mid <= len; mid <<= 1) {
        for (int i = 0; i < len; i += mid) {
            for (int j = i; j < i + (mid >> 1); j++) {
                a[j] += a[j + (mid >> 1)];
			}
		}
	}
	return ;
}
void ifwt_and(vector<ll> &a, int len) {
    for (int mid = 2; mid <= len; mid <<= 1) {
        for (int i = 0; i < len; i += mid) {
            for (int j = i; j < i + (mid >> 1); j++) {
                a[j] -= a[j + (mid >> 1)];
			}
		}
	}
	return ;
}
```

#### XOR

```C++
void fwt_xor(vector<ll> &a, int len) {
    for (int mid = 2; mid <= len; mid <<= 1) {
        for (int i = 0; i < len; i += mid) {
            for (int j = i; j < i + (mid >> 1); j++) {
                ll x = a[j], y = a[j + (mid >> 1)];
                a[j] = x + y, a[j + (mid >> 1)] = x - y;
            }
		}
	}
	return ;
}
void ifwt_xor(vector<ll> &a, int len) {
    for (int mid = 2; mid <= len; mid <<= 1) {
        for (int i = 0; i < len; i += mid) {
            for (int j = i; j < i + (mid >> 1); j++) {
                ll x = a[j], y = a[j + (mid >> 1)];
                a[j] = (x + y) >> 1, a[j + (mid >> 1)] = (x - y) >> 1;
            }
		}
	}
	return ;
}
```

## Mess

### BSGS

$$
a^x=b(modp)
$$

> BSGS（Baby-Step-Giant-Step），即大步小步算法。常用于求解离散对数问题，即上面引入中题目的方程$a^x=b(modp)$ ，它可以在$O(\sqrt{p})$的时间复杂度下完成求解，不过要求$a$与$mod$互质.
>
> 来看看 BSGS 是如何做到的：
>
> 首先令$x = a^{A\lceil{\sqrt{p}}\rceil}-B$，其中$0\leq{A,B}\leq{\lceil{\sqrt{p}}\rceil}$ ，将其代入方程得$a^{A\lceil{\sqrt{p}}\rceil-B}=b(modp)$ ，进行指数分解和移项可以得到 $a^{A\lceil{\sqrt{p}}\rceil}=b\cdot{a^B(modp)}$。
>
> 然后分析一下这个方程，可以发现我们已知参数为$a$和$b$，那么我们就可以通过枚举$B$来计算同余方程右端的所有取值，将它们存入哈希表中（方便后面查找）。
>
> 接着对于方程左端，同样可以通过枚举$A$得到所有值，如果在得到的这些值中找到与右端的值相同的 $A$，就可以通过 $x = a^{A\lceil{\sqrt{p}}\rceil}-B$得到方程的解了。
>
> 这样由于 A 和 B 的范围为 $0\leq{A,B}\leq{\lceil{\sqrt{p}}\rceil}$，枚举 A 和 B 的时间复杂度就为$O(\sqrt{p})$ 。
>
> （对于引入中的问题，只要在枚举 A 的过程中发现值与右端中某个值相等，那么就可以直接返回 $x$，因为在 A 的增大过程中， 是逐渐增大的，因此第一次找到相等情况的就是答案）

```cpp
ll bsgs(ll a, ll b, ll mod){
    map<ll, ll> mp;
    ll cur = 1, t = sqrt(mod) + 1;
    for(int B = 1; B <= t; B++){
        cur = cur * a % mod;
        mp[b * cur % mod] = B;
    }
    ll now = cur;
    for(int A = 1; A <= t; A++){
        if(mp[now]) return (ll)A * t - mp[now];
        now = now * cur % mod;
    }
    return -1;
}
```

### 小数循环节

```c++
int Circle(int tmp) {
    while(tmp % 2 == 0) tmp /= 2;
	while(tmp % 5 == 0) tmp /= 5;
	int len = 0, e = 1;
	if(tmp == 1) return -1;
	while(1) {
		e = e * 10 % tmp;
		if(e == 1) break;
		len++;
	}
	return len;
}
```

### SG Func

**SG函数**是用于解决**博弈论**中**公平组合游戏**（**I**mpartial **C**ombinatorial **G**ames，**ICG**）问题的一种方法。

------

所谓ICG，应当满足以下几条性质：

1. 有两名玩家
2. 两名玩家轮流操作，在一个有限集合内任选一个进行操作，改变游戏当前局面
3. 一个局面的合法操作，只取决于游戏局面本身且固定存在，与玩家次序或者任何其它因素无关
4. 无法操作者，即操作集合为空，输掉游戏，另一方获胜[[1\]](https://zhuanlan.zhihu.com/p/257013159#ref_1)

最简单的ICG当然就是经典的Nim游戏了：

> 地上有 ![[公式]](https://www.zhihu.com/equation?tex=n) 堆石子，每堆石子数量可能不同，两人轮流操作，每人每次可从任意一堆石子里取走任意多枚石子，可以取完，不能不取，无石子可取者输掉游戏，问是否存在先手必胜的策略。

之后我们会多次以Nim游戏举例。

------

对于ICG，我们定义一个函数 ![[公式]](https://www.zhihu.com/equation?tex=sg%28x%29) ，令 ![[公式]](https://www.zhihu.com/equation?tex=sg%28x%29%3A%3D%5Cmathrm%7Bmex%7D%28%5C%7Bsg%28y%29%7Cx%5Crightarrow+y%5C%7D%29) 。其中， ![[公式]](https://www.zhihu.com/equation?tex=x) 和 ![[公式]](https://www.zhihu.com/equation?tex=y) 都表示某种**状态**， ![[公式]](https://www.zhihu.com/equation?tex=x%5Crightarrow+y) 在这里表示 ![[公式]](https://www.zhihu.com/equation?tex=x) 状态可以通过**一次**操作**达到** ![[公式]](https://www.zhihu.com/equation?tex=y) 状态， ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathrm%7Bmex%7D) 表示一个集合中未出现的最小自然数（例如 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathrm%7Bmex%7D%28%5C%7B0%2C1%2C3%5C%7D%29%3D2) ，经常打[cf](https://link.zhihu.com/?target=http%3A//codeforces.com/)的人应该对这个函数比较熟悉）。如果 ![[公式]](https://www.zhihu.com/equation?tex=sg%28x%29%3Dn) ，说明从当前状态可以转移到 ![[公式]](https://www.zhihu.com/equation?tex=sg) 为 ![[公式]](https://www.zhihu.com/equation?tex=0%2C1%2C%5Cdots%2Cn-1) 的状态。

我们称令 ![[公式]](https://www.zhihu.com/equation?tex=sg%28x%29%3D0) 的状态 ![[公式]](https://www.zhihu.com/equation?tex=x) 为**必败态**。显然，如果 ![[公式]](https://www.zhihu.com/equation?tex=x) 是**无法操作**的状态（即 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bsg%28y%29%7Cx%5Crightarrow+y%5C%7D%3D%5Cvarnothing) ，在Nim游戏中是没有石子可取的状态），则必有 ![[公式]](https://www.zhihu.com/equation?tex=sg%28x%29%3D0) ，这时当前玩家输掉游戏；若不然，则该状态只能转移到 ![[公式]](https://www.zhihu.com/equation?tex=sg) 不为 ![[公式]](https://www.zhihu.com/equation?tex=0) 的状态，那么对手立刻又可以转移到 ![[公式]](https://www.zhihu.com/equation?tex=sg) 为 ![[公式]](https://www.zhihu.com/equation?tex=0+) 的状态，这样进行有限次后一定会陷入无法操作的状态输掉比赛。

相反，如果 ![[公式]](https://www.zhihu.com/equation?tex=sg%28x%29%5Cne0) ，则称 ![[公式]](https://www.zhihu.com/equation?tex=x) 为**必胜态**，说明此时存在策略使自己必定取胜（即每次轮到自己都转移到 ![[公式]](https://www.zhihu.com/equation?tex=sg) 为 ![[公式]](https://www.zhihu.com/equation?tex=0) 的状态）。

对于单堆的Nim游戏，我们很容易计算其 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值。设 ![[公式]](https://www.zhihu.com/equation?tex=sg%28m%29) 表示剩余石子数为 ![[公式]](https://www.zhihu.com/equation?tex=m) 的状态的 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值，则显然 ![[公式]](https://www.zhihu.com/equation?tex=sg%280%29%3D0) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=sg%281%29%3D1) ，![[公式]](https://www.zhihu.com/equation?tex=sg%282%29%3D2) ……归纳可证 ![[公式]](https://www.zhihu.com/equation?tex=sg%28n%29%3Dn) 。因此，石子数不为0都是必胜态。

对于双堆的Nim游戏，设 ![[公式]](https://www.zhihu.com/equation?tex=sg%28n%2Cm%29) 表示剩余石子数分别为 ![[公式]](https://www.zhihu.com/equation?tex=n) 和 ![[公式]](https://www.zhihu.com/equation?tex=m) 的状态的 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值，则 ![[公式]](https://www.zhihu.com/equation?tex=sg%280%2C0%29%3D0) ，并且 ![[公式]](https://www.zhihu.com/equation?tex=sg%28n%2C0%29%3Dsg%280%2Cn%29%3Dn) 。我们发现 ![[公式]](https://www.zhihu.com/equation?tex=sg%281%2C1%29%3D0) ，而 ![[公式]](https://www.zhihu.com/equation?tex=sg%281%2C2%29%3Dsg%282%2C1%29%3D1) ，![[公式]](https://www.zhihu.com/equation?tex=sg%282%2C2%29%3D0) ……继续探索可以猜测，当 ![[公式]](https://www.zhihu.com/equation?tex=n%3Dm) 时，![[公式]](https://www.zhihu.com/equation?tex=sg%28n%2Cm%29%3D0) ，否则 ![[公式]](https://www.zhihu.com/equation?tex=sg%28n%2Cm%29%5Cne0) 。这可以归纳证明。

当堆数越来越多时，这样盲目探索显得很困难。所幸，我们有**SG定理**。

------

定义若干个ICG的**组合**为这样一个游戏：每位玩家每个回合在这若干个ICG（称为**子游戏**）中选择一个，对它进行一次操作；当所有子游戏都无法进行操作时，当前玩家输掉游戏。记两个ICG的状态分别为 ![[公式]](https://www.zhihu.com/equation?tex=x) 和 ![[公式]](https://www.zhihu.com/equation?tex=y) ，则记这两个ICG的组合的状态为 ![[公式]](https://www.zhihu.com/equation?tex=x%2By) 。

**Sprague-Grundy定理**（两个游戏的情形）：

> 设 ![[公式]](https://www.zhihu.com/equation?tex=x) 和 ![[公式]](https://www.zhihu.com/equation?tex=y) 为ICG的状态，则： ![[公式]](https://www.zhihu.com/equation?tex=sg%28x%2By%29%3Dsg%28x%29%5Coplus+sg%28y%29)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Coplus) 表示异或。

这个定理还是很神奇的，这里就不展开篇幅证明了，[这里](https://www.zhihu.com/question/51290443)有很多证明。可以这样感性地理解：如果两个子游戏的 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值异或**不为** ![[公式]](https://www.zhihu.com/equation?tex=0) ，那么一定可以转移到一个 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值异或和**为** ![[公式]](https://www.zhihu.com/equation?tex=0) 的状态（将 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值较大者转移到与较小者的 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值相同的状态就可以了，这一定能做到）；相反，如果两个子游戏的 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值异或**为** ![[公式]](https://www.zhihu.com/equation?tex=0)，则只能转移到 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值异或和**不为** ![[公式]](https://www.zhihu.com/equation?tex=0) 的状态——这正好对应了必胜态和必败态。

这个定理很容易进行**推广**：

> 设 ![[公式]](https://www.zhihu.com/equation?tex=x_1%2Cx_2%2C%5Cdots%2Cx_n) 为ICG的状态，则： ![[公式]](https://www.zhihu.com/equation?tex=sg%28x_1%2Bx_2%2B%5Cdots%2Bx_n%29%3Dsg%28x_1%29%5Coplus+sg%28x_2%29%5Coplus%5Cdots%5Coplus+sg%28x_n%29)

利用SG定理，我们就可以轻松地把单一游戏的情况推广到多个子游戏组合的情况。例如Nim游戏，就是在 ![[公式]](https://www.zhihu.com/equation?tex=n) 堆石子的异或和为 ![[公式]](https://www.zhihu.com/equation?tex=0) 时必败，否则必胜。

------

举一个例题：

#### Eg1

> 小 E 与小 W 进行一项名为 `E&D` 游戏。
> 游戏的规则如下：桌子上有 ![[公式]](https://www.zhihu.com/equation?tex=2n) 堆石子，编号为 ![[公式]](https://www.zhihu.com/equation?tex=1+%5Csim+2n) 。其中，为了方便起见，我们将第 ![[公式]](https://www.zhihu.com/equation?tex=2k-1) 堆与第 ![[公式]](https://www.zhihu.com/equation?tex=2k) 堆（ ![[公式]](https://www.zhihu.com/equation?tex=1+%5Cle+k+%5Cle+n) ）视为同一组。第 ![[公式]](https://www.zhihu.com/equation?tex=i) 堆的石子个数用一个正整数 ![[公式]](https://www.zhihu.com/equation?tex=S_i) 表示。
> 一次分割操作指的是，从桌子上任取一堆石子，将其移走。然后分割它同一组的另一堆石子，从中取出若干个石子放在被移走的位置，组成新的一堆。操作完成后，所有堆的石子数必须保证大于 ![[公式]](https://www.zhihu.com/equation?tex=0) 。显然，被分割的一堆的石子数至少要为 ![[公式]](https://www.zhihu.com/equation?tex=2) 。两个人轮流进行分割操作。如果轮到某人进行操作时，所有堆的石子数均为 ![[公式]](https://www.zhihu.com/equation?tex=1) ，则此时没有石子可以操作，判此人输掉比赛。
> 小 E 进行第一次分割。他想知道，是否存在某种策略使得他一定能战胜小 W。因此，他求助于小 F，也就是你，请你告诉他是否存在必胜策略。例如，假设初始时桌子上有 ![[公式]](https://www.zhihu.com/equation?tex=4) 堆石子，数量分别为 ![[公式]](https://www.zhihu.com/equation?tex=1%2C2%2C3%2C1) 。小 E 可以选择移走第 ![[公式]](https://www.zhihu.com/equation?tex=1) 堆，然后将第 ![[公式]](https://www.zhihu.com/equation?tex=2) 堆分割（只能分出 ![[公式]](https://www.zhihu.com/equation?tex=1) 个石子）。接下来，小 W 只能选择移走第 ![[公式]](https://www.zhihu.com/equation?tex=4) 堆，然后将第 ![[公式]](https://www.zhihu.com/equation?tex=3) 堆分割为 ![[公式]](https://www.zhihu.com/equation?tex=1) 和 ![[公式]](https://www.zhihu.com/equation?tex=2) 。最后轮到小 E，他只能移走后两堆中数量为 ![[公式]](https://www.zhihu.com/equation?tex=1) 的一堆，将另一堆分割为 ![[公式]](https://www.zhihu.com/equation?tex=1) 和 ![[公式]](https://www.zhihu.com/equation?tex=1) 。这样，轮到小 W 时，所有堆的数量均为 ![[公式]](https://www.zhihu.com/equation?tex=1) ，则他输掉了比赛。故小 E 存在必胜策略。

很显然这是一个ICG，且可以看作 ![[公式]](https://www.zhihu.com/equation?tex=n) 个子游戏的组合，显然每一组两堆石子组成最小单位。我们直接打表列出 ![[公式]](https://www.zhihu.com/equation?tex=sg) 值（按照定义计算，可以写程序或手算）：

![img](https://pic4.zhimg.com/80/v2-7cb5ba12b143387b381c6c2405c74657_720w.jpg)

这张表显然很有规律，例如当 ![[公式]](https://www.zhihu.com/equation?tex=a) 和 ![[公式]](https://www.zhihu.com/equation?tex=b) 是奇数时 ![[公式]](https://www.zhihu.com/equation?tex=s%28a%2Cb%29%3D0) ，还有 ![[公式]](https://www.zhihu.com/equation?tex=sg%28i%2Cj%29%3Dsg%28j%2Ci%29) 等等。还很容易发现，偶数列（或行）都是几个几个一组的。在此基础上进一步观察，可以发现第 ![[公式]](https://www.zhihu.com/equation?tex=2n) 列的值与第 ![[公式]](https://www.zhihu.com/equation?tex=n) 列是有关联的。

![img](https://pic4.zhimg.com/80/v2-1ea2c34a3d82d87c16ccf2e89c33c6cb_720w.jpg)

总结成公式就是：当 ![[公式]](https://www.zhihu.com/equation?tex=a) 为偶数时， ![[公式]](https://www.zhihu.com/equation?tex=sg%28a%2Cb%29%3Dsg%28%5Cfrac%7Ba%7D%7B2%7D%2C%5Cleft%5Clfloor%5Cfrac%7Bb-1%7D%7B2%7D%5Cright%5Crfloor%2B1%29%2B1) 。于是可以递归求解，代码如下：

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

int count(ll n) {
    int cnt = 0;
    while (n % 2 == 0)
        n /= 2, cnt++;
    return cnt;
}
int sg(ll a, ll b) {
    if (a % 2 && b % 2)
        return 0;
    else if (a % 2 == 0)
        return sg(a / 2, (b - 1) / 2 + 1) + 1;
    else
        return sg(b, a);
}
int main() {
    ll t, n, x, y;
    cin >> t;
    while (t--) {
        int res = 0;
        cin >> n;
        for (int i = 0; i < n / 2; ++i) {
            cin >> x >> y;
            res ^= sg(x, y);
        }
        cout << (res ? "YES" : "NO") << endl;
    }
    return 0;
}
```

#### Eg2

有$n$个格子排成一排，每个颜色为黑色或白色，第$i$个格子的颜色为$c_i$，如果$c_i=b$，则第$i$个格子为黑色，如果$c_i=w$，则第$i$个颜色为白色。
 牛牛和牛妹在这$n$个格子上玩游戏，两个人轮流操作，牛妹先手，每次操作可以为以下类型之一：

* 选择两个整数$i,j(1≤i<j≤n)$满足第$j$个石子为白色，同时翻转第$i$个格子和第$j$个格子的颜色。

* 翻转第$1$个格子的颜色（第$1$个格子的颜色为白色时才能进行此操作）。

   不能操作的人输，牛牛和牛妹都非常聪明，牛牛和牛妹谁会赢？

> Input
>
> bwbb
>
> wbbb
>
> ----------------------------------------------
>
> Output
>
> No
>
> Yes

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

	vector<int> sg(1 << 12);

	auto Pre = [&]() {
		sg[0] = 0;
		for(int i = 1;i < 1 << 11;i++) {
			if(i & 1) {
				if(sg[i ^ 1] == 0) {
					sg[i] = 1;
				}
			}
			for(int j = 1;j < 11;j++) {
				if(i >> j & 1) {
					for(int k = 0;k < j;k++) {
						if(sg[i ^ (1 << j) ^ (1 << k)] == 0) {
							sg[i] = 1;
						}
					}
				}
			}
		}
	};

	Pre();

	int t;
	cin >> t;
	while(t--) {
		int n;
		cin >> n;
		string s;
		cin >> s;
		int x = 0;
		for(int i = 0;i < n;i++) {
			if(s[i] == 'w') {
				x |= (1 << i);
			}
		}
		if(sg[x]) cout << "Yes" << '\n';
		else cout << "No" << '\n';
	}

    return 0;
}
```

我们并没有严格地证明观察出结论，但这就是这类题的通常做法：**打表找规律**。严格证明需要花的时间，对ACM来说太奢侈了。以下这个`mex`函数可能会经常用于打表：

```cpp
int mex(auto v) // v可以是vector、set等容器 
{
    unordered_set<int> S;
    for (auto e : v)
        S.insert(e);
    for (int i = 0;; ++i)
        if (S.find(i) == S.end())
            return i;
}
```

# String 字符串

## KMP

```c++
const int maxn = 1e5 + 7;
vector<int> ne(maxn, 0);
string mode, text;
void getNext() {
	int j = 0, k = -1;
	ne = vector<int>(maxn, 0);
	ne[0] = -1;
	while(j < mode.length()) {	
		if(k == -1 || mode[j] == mode[k]) ne[++j] = ++k;
       	else k = ne[k];
	}
}
int KMP_Count() {   // 求匹配次数
	int mode_len = mode.length();
	int text_len = text.length();
	getNext();
	int i = 0, j = 0, ans = 0;
	for(;i < text_len;i++) {
		while(j > 0 && text[i] != mode[j]) j = ne[j];
		if(text[i] == mode[j]) j++;
		if(j == mode_len) {
			ans++;
			j = ne[j];
		}
	}
	return ans;
}
int KMP_Index() {   
	//求第一次匹配到的位置， 当然你也可以find()
	int mode_len = mode.length();
	int text_len = text.length();
	int i = 0, j = 0;
	getNext();
	while(i < text_len && j < mode_len) {
		if(j == -1 || text[i] == mode[j]) i++, j++;	
		else j=ne[j];
	}
	if(j >= mode_len) return i - mode_len;
	else return -1;
}
```

## 扩展KMP

## Manacher

```c++
void Manacher(string ss){
	vector<int > id(1000100, 0);
	int r = 0, mid = 0;
	string s = "$#";
	for(int i = 0;i < ss.length();i++) s += ss[i], s += '#';
	for(int i = 1;i <= s.length();i++) {
		id[i] = r > i ? min(id[2 * mid - i], r - i) : 1;
		while(s[i + id[i]] == s[i - id[i]]) id[i]++;
		if(id[i] > r - i){
			r = id[i] + i;
			mid = i;
		}
	}
    // 对称下标
	int maxId = max_element(id.begin(), id.end()) - id.begin(); 
    // 最长回文串长度
	int maxLen = *max_element(id.begin(), id.end()) - 1;
    // 最长回文串
	string maxPalindrome = "";
	for(int i = maxId - maxLen;i <= maxId + maxLen;i++) if(s[i] != '#') maxPalindrome += s[i];
    return ;
}

```

## Hash

``` c++
using ull = unsigned long long;

const ull p = 131;
const int maxn = 1e6 + 7;

struct StringHash {
    string s;
    vector<ull> ha, p_;
    //string in interval [1, n]
    StringHash(const string& s_) : s(s_) {
        ha.resize(s_.size() << 1);
        p_.resize(s_.size() << 1, 1);
        int n = s_.size();
        for(int i = 1;i <= n;i++) {
            ha[i] = ha[i - 1] * p + s[i - 1];
            p_[i] = p_[i - 1] * p;
        }
    }
    ull get(int l, int r) {
        if(r < l) return 0;
        return ha[r] - ha[l - 1] * p_[r - l + 1];
    }
};
```

## 字典树

### 01字典树（求最大异或）

```c++
const int maxn = 2e5 + 7;
int tot = 1;

vector<vector<int> > tr(maxn << 5, vector<int>(2, 0));
vector<int> vis(maxn << 5, 0);
	
void insert(int x, int v) {
	int rt = 1;
	for(int i = 30;i >= 0;i--) {
		int now = (x >> i) & 1;
		if(!tr[rt][now]) tr[rt][now] = ++tot;
		rt = tr[rt][now];
		//vis[rt] += v;// mark suffix
	}
	//do mark
	return ;
}

int query(int x) {
	int rt = 1, ret = 0;
	x = ~x;
	for(int i = 30;i >= 0;i--) {
		int now = (x >> i) & 1;
		ret <<= 1;
		if(tr[rt][now] && vis[tr[rt][now]]) {
			ret |= 1;
			rt = tr[rt][now];
		}
		else rt = tr[rt][now ^ 1];
	}
	return ret;
}

```

## AC自动机

```c++
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

const int maxn = 2e6 + 7;

char s[maxn], T[maxn];
int n, cnt, vis[200051], ans, in[maxn], Map[maxn];
//求每个模式串在文本串中出现的次数

struct kkk {
    int son[26], fail, flag, ans;
    void clear() {
        memset(son, 0, sizeof(son));
        fail = flag = ans = 0;
    }
} trie[maxn];
queue<int>q;

void insert(char *s, int num) {
    int u = 1, len = strlen(s);
    for (int i = 0; i < len; i++) {
        int v = s[i] - 'a';
        if (!trie[u].son[v])
            trie[u].son[v] = ++cnt;
        u = trie[u].son[v];
    }
    if (!trie[u].flag)
        trie[u].flag = num;
    Map[num] = trie[u].flag;
}

void getFail() {
    for (int i = 0; i < 26; i++)
        trie[0].son[i] = 1;
    q.push(1);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        int Fail = trie[u].fail;
        for (int i = 0; i < 26; i++) {
            int v = trie[u].son[i];
            if (!v) {
                trie[u].son[i] = trie[Fail].son[i];
                continue;
            }
            trie[v].fail = trie[Fail].son[i];
            in[trie[v].fail]++;
            q.push(v);
        }
    }
}

void topu() {
    for (int i = 1; i <= cnt; i++)
        if (in[i] == 0)
            q.push(i);

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        vis[trie[u].flag] = trie[u].ans;
        int v = trie[u].fail;
        in[v]--;
        trie[v].ans += trie[u].ans;
        if (in[v] == 0)
            q.push(v);
    }
}
void query(char *s) {
    int u = 1, len = strlen(s);

    for (int i = 0; i < len; i++)
        u = trie[u].son[s[i] - 'a'], trie[u].ans++;
}
int main() {
    scanf("%d", &n);
    cnt = 1;

    for (int i = 1; i <= n; i++) {
        scanf("%s", s);
        insert(s, i);
    }

    getFail();
    scanf("%s", T);
    query(T);
    topu();

    for (int i = 1; i <= n; i++)
        printf("%d\n", vis[Map[i]]);
    return 0;
}
```

## 后缀数组

##  后缀自动机

# Geometry 计算几何

## 点到线段的距离

```c++
double patch_line_point(double x , double y , double line_x1 , double line_y1 , double line_x2 , double line_y2) {
	double x1 = line_x1, y1 = line_y1, x2 = line_x2, y2 = line_y2, x3 = x, y3 = y;  
	double px = x2 - x1;
    double py = y2 - y1;
    double som = px * px + py * py;
    double u =  ((x3 - x1) * px + (y3 - y1) * py) / som;
    if (u > 1) u = 1;
    if (u < 0) u = 0;
    double xx = x1 + u * px;
    double yy = y1 + u * py;
	double dx = xx - x3;
    double dy = yy - y3;    
    double dist = sqrt(dx * dx + dy * dy);
    return dist;
}
```

## 四面体体积计算（已知四个点的坐标）

```c++
struct point {
    double x, y, z;
} p[5];
double line2(point a, point b) {
    return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)) + ((a.z - b.z) * (a.z - b.z));
}
void work() {
    for (int i = 1; i <= 4; i++) cin >> p[i].x >> p[i].y >> p[i].z;

    double p12 = line2(p[1], p[2]);
    double p13 = line2(p[1], p[3]);
    double p14 = line2(p[1], p[4]);
    double p23 = line2(p[2], p[3]);
    double p24 = line2(p[2], p[4]);
    double p34 = line2(p[3], p[4]);
    double res1 = p12 * p34 * (p13 + p14 + p23 + p24 - p12 - p34);
    double res2 = p14 * p23 * (p12 + p13 + p24 + p34 - p14 - p23);
    double res3 = p13 * p24 * (p12 + p14 + p23 + p34 - p13 - p24);
    double res4 = (p12 * p13 * p23) + (p13 * p14 * p34) + (p12 * p14 * p24) + (p24 * p34 * p23);
    double v = sqrt(res1 + res2 + res3 - res4) / 12;
    printf("%.12f\n", v);
}
```

## 判断线段相交

```c++
int CrossProduct(ll x1, ll y1, ll x2, ll y2) {
    ll xx = x1 * y2;
    ll yy = y1 * x2;
    if (xx == yy) return 0;
    return xx > yy ? 1 : -1;
}

bool cek(ll x1, ll y1, ll x2, ll y2, ll x3, ll y3, ll x4, ll y4) {//两条线段的起点和终点
	return CrossProduct(x2 - x1, y2 - y1, x3 - x1, y3 - y1) * CrossProduct(x2 - x1, y2 - y1, x4 - x1, y4 - y1) <= 0 &&
           CrossProduct(x4 - x3, y4 - y3, x1 - x3, y1 - y3) * CrossProduct(x4 - x3, y4 - y3, x2 - x3, y2 - y3) <= 0;
}
```

# STL模板库

## Vector

* 初始化

```c++
vector<int > a(n + 1, 0);
vector<vector<int > > a(n, vector<int>(m, 0));
vector<vector<vector<int> > > dp(n, vector<vector<int> >(n, vector<int>(n, 0)));
```

* 构建右值vector

``` c++
vector<int >(n + 1, 0);
vector<vector<int > >(n + 1, vector<int > (n + 1, 0));
vector<vector<vector<int> > > (n, vector<vector<int> >(n, vector<int>(n, 0)));
```

* 读入

```c++
for(auto &it : a) cin >> it;
for(int i = 1;i <= n;i++) cin >> a[i];
```

​	已经重载了的operator

```c++
>  <  >=  <=  ==  =  != [] 
```

* 常用成员方法

```c++
unique(iterator1, iterator2);//去重，返回去重后的尾迭代器
iterator erase( iterator loc );
iterator erase( iterator start, iterator end );
sort(iterator1, iterator2, less<int>() / greater<int>());
push_back(const TYPE &val);
emplace_back(const TYPE &val);
pop_back();
back();
front();
clear();
size();
swap();//交换两个vector
lower_bound(iterator1, iterator2, val); // return iterator;
upper_bound(iterator1, iterator2, val); // return iterator;
reverse(iterator1, iterator2);
max_element(iterator1, iterator2); // return iterator
min_element(iterator1, iterator2); // return iterator
accumulate(iterator1, iterator2, init) // return sum + init
void resize( size_type size, TYPE val );
```

- 在指定位置loc前插入值为val的元素,返回指向这个元素的迭代器
- 在指定位置loc前插入num个值为val的元素 
- 在指定位置loc前插入区间[start, end)的所有元素

```c++
iterator insert( iterator loc, const TYPE &val );
void insert( iterator loc, size_type num, const TYPE &val );
void insert( iterator loc, input_iterator start, input_iterator end );
```

## Stack

​	常用成员方法

```c++
stack<int> s;

s.pop();
s.push();
s.top();
s.empty();
s.size();
```

## String

​	初始化

```c++
string s(str);
string s(n, '0');
string s(str)
```

​	常用成员方法

```c++
string s;

s.push_back()//Or not "+="
s.size() / s.length();
s.insert(id(iterator), char);
s.erase(iterator);
s.find(const char);
s.sort();
s.substr();
s.reverse();
tolower() / toupper() : return char;
c_str(); return *char;
replace(begin, end, string2);
```

​	关于返回值

```c++
string::npos;  //find()函数失配时的返回值
```

​	string与int之间的转化(#include<sstream>)

```c++
int to_int(string s) {
	stringstream Stream;
	int ret;
	Stream << s;
	Stream >> ret;
	Stream.clear();
	return ret;
}
string to_String(int x) {
	stringstream Stream;
	string ret;
	Stream << x;
	Stream >> ret;
	Stream.clear();
	return ret;
}
```

## Bitset

​	Operators

- != 返回真如果两个bitset不相等。 
- == 返回真如果两个bitset相等。 
- &= 完成两个bitset间的与运算。 
- ^= 完成两个bitset间的异或运算。 
- |= 完成两个 
- ~ 反置bitset (和调用 [flip()](#flip)类似) 
- <<= 把bitset向左移动 
- \>>= 把bitset向右移动 
- [] 返回第x个位的引用

``` c++
&   ^   |   >>   <<   <<=   >>= 
!=  ==  &=  ^=  |=  ~  []
```

​	初始化

```c++
bitset<n> bit(0); // 创建一个n位的bitset，初始化为0
bitset<n> bit(s); // s 可以为unsigned或string
```

​	常用成员方法

```c++
any();     // 是否存在为1的二进制位
count();   // 二进制中1的个数
size();    // 二进制位的个数
set();     // 把所有位都制为1
reset();   // 把所有位都制位0
flip();    // 把所有位按位取反
to_ulong();// 返回unsigned long long 
Stream << bit;  // 把位集输入到Stream流中，建议输出string流
```

## Deque

​	Operators

```c++
[]
```

​    常用成员方法

```c++
deque<int> q;

q.back();
q.front();
q.pop_back();
q.pop_front();
q.push_back();
q.push_front();
q.empty();
q.clear();
q.size();
q.resize(size_type num, TYPE val);

sort();
```

## Queue / Priority_queue

​    常用成员方法

```c++
queue<int> q;

q.back();
q.front();
q.empty();
q.pop();
q.push();
q.size();
```

## Map / Unordered_map/Multimap

```cpp
map<int. int> mp;

mp.lower_bound(x);
mp.upper_bound(x);
mp.count(x);
mp.erase(Iter);
mp.find(x);
mp.insert(x);
mp.size();
```

## Set / Unordered_set/Multiset

```c++
set<int> st;

st.lower_bound(x);
st.upper_bound(x);
st.count(x);
st.erase(Iter);
st.find(x);
st.insert(x);
st.size();
```

## Utility

* pair

```c++
pair<Type, Type> p;
```

## Algorithm

```c++
nth_element(a.begin(), a.begin() + k, a.end(), less/greater);
stable_sort(a.begin(), a.end());
```

## Iterator

# Mess

## Q_read/Q_write

```c++
inline int qr(){
    int x = 0, f = 1; char c = getchar();
    while(!isdigit(c)) { 
        if(c == '-') f = -f;
        c = getchar();
    }
    while(isdigit(c)) { 
        x = (x << 1) + (x << 3) + (c ^ 48); 
        c = getchar(); 
    }
    return x * f;
}
inline void qw(int a) {
    if(a < 0) putchar('-'), a = -a;
    if(a > 9) qw(a / 10);
    putchar((a % 10) ^ 48);
}
```

## DEBUG

``` c++
#define dbg(x) cerr << #x << "=" << x << '\n'

template<class t, class u>
ostream &operator<<(ostream &os, const pair<t, u> &p) {
    return os << "(" << p.first << ", " << p.second << ")";
}

template<class t>
ostream &operator<<(ostream &os, const vector<t> &v) {
    os << "[" << (*v.begin());
    for (int i = 1;i < v.size();i++) os << ", " << v[i];
    return os << "]";
}
```

## Mint

```c++
constexpr int P = 1000000007;
using ll = long long;
// assume -P <= x < 2P
int norm(int x) {if(x < 0)x += P;if (x >= P)x -= P;return x;}
template<class T>T power(T a, int b){T res = 1;for (; b; b /= 2, a *= a)if (b & 1)res *= a;return res;}
struct Mint {
    int x;Mint(int x = 0) : x(norm(x)){}
    int val() const {return x;}
    Mint operator-() const {return Mint(norm(P - x));}
    Mint inv() const {assert(x != 0);return power(*this, P - 2);}
    Mint &operator*=(const Mint &rhs) { x = ll(x) * rhs.x % P; return *this;}
    Mint &operator+=(const Mint &rhs) { x = norm(x + rhs.x); return *this;}
    Mint &operator-=(const Mint &rhs) { x = norm(x - rhs.x); return *this;}
    Mint &operator/=(const Mint &rhs) {return *this *= rhs.inv();}
    friend Mint operator*(const Mint &lhs, const Mint &rhs) {Mint res = lhs; res *= rhs; return res;}
    friend Mint operator+(const Mint &lhs, const Mint &rhs) {Mint res = lhs; res += rhs; return res;}
    friend Mint operator-(const Mint &lhs, const Mint &rhs) {Mint res = lhs; res -= rhs; return res;}
    friend Mint operator/(const Mint &lhs, const Mint &rhs) {Mint res = lhs; res /= rhs; return res;}
};
```

## Unordered_map/set Hash

```c++
struct haha {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
```

## Random

### 随机数生成

```c++
mt19937 rng(chrono::system_clock::now().time_since_epoch().count());
```

### 随机打乱

```c++
random_shuffle(x.begin(), x.end());
```

## Duipai

```c++
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

	int T = 10005;
	while(T) {
		string name = "23481B";

//		if(system("fc out.txt STD_out.txt")) break;		

//		system(("data.exe > " + name + ".in").c_str());	
		system("data.exe > data.txt");	
		//生成数据到data.txt
		system((name + ".exe < data.txt > out.txt").c_str());		
		//你的代码运行结果输出到out.txt
		system((name + "_std.exe < data.txt > STD_out.txt").c_str());
		//标称代码运行结果输出到STD_out.txt
		if(system("fc out.txt STD_out.txt")) break;		
		//将两个输出文件进行对比，若不同则break

		T -= 1;
	}
	if(T) cout << "Error" << '\n';
	else cout << "Pass" << '\n';

    return 0;
}
```

## Construct Data

### Random Tree

```cpp
#include<bits/stdc++.h>

using namespace std;

int n, cnt,fa[100015];

mt19937 rng(chrono::system_clock::now().time_since_epoch().count());

int find(int x){
	return x == fa[x] ? x : fa[x] = find(fa[x]);
}

int main() {
	n = rng() % 10 + 1;
	cout << n << '\n';
	for(int i = 1; i <= n; i++) fa[i] = i;
	while(cnt < n - 1){
		unsigned x = rng(), y = rng();
		x = x % n + 1;
		y = y % n + 1;
		int x1 = find(x), y1 = find(y);
		if(x1 != y1) {
			fa[x1] = y1;
			cnt++;
			if(x > y) swap(x, y);
			cout << x << ' ' << y << '\n';
		}
	}
	return 0;
}
```

# Brute Force

## 枚举二进制的子集

``` c++
int x; cin >> x;
for(int i = x;i;i = (i - 1) & x) cout << i << '\n';
```

## 二分

### 二分答案

#### 01分数规划

$$
给出n个物品，每个物品有两个属性a_i和b_i，选择k个物品，询问\frac{\sum{a_i}}{\sum{b_i}}的最大值
$$


$$
\frac{\sum{a_i}}{\sum{b_i}}\geq x =\sum{a_i}-x\sum{b_i} \geq 0
$$

* 对x二分答案即可求得最大值
* f(x)为a[i] - x * b[i]的值

```c++
double l = 0, r = 1e10;

function<bool(double)> cek = [&](double x) {
    vector<double> tmp(n + 1, 0);
	for(int i = 1;i <= n;i++) tmp[i] = a[i] - x * b[i];
	sort(tmp.begin() + 1, tmp.end(), [](double x, double y) {
		return x > y;
	});
	double ret = 0;
	for(int i = 1;i <= k;i++) ret += tmp[i];
	return ret >= 0;
};

while(fabs(l - r) > eps) {
	double mid = (l + r) / 2.0;
	if(cek(mid)) l = mid;
	else r = mid;
}
```

#### 树状数组上二分寻找第k大

### 整体二分

## 根号算法



