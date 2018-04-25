
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <queue>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <time.h>

const int MAX_NODES = 1632803 + 1;
const int MAX_EDGES = 30622564 + 1;
int no_of_nodes;
int no_of_edges;
const int THREADS_PER_BLOCK = 512;

struct Node {
	int start;
	int no_of_edges;
};

// explore neighbours of every vertex from the current level
__global__ void bfs(Node* dev_graph,
	bool* dev_frontier,
	bool* dev_update,
	bool* dev_visited,
	int* dev_edge_list,
	int* dev_cost,
	int* dev_MAX_NODES)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < *dev_MAX_NODES && dev_frontier[tid]) {
		dev_frontier[tid] = 0;
		for (int i = 0; i < dev_graph[tid].no_of_edges; ++i) {
			int nid = dev_edge_list[i + dev_graph[tid].start];
			if (!dev_visited[nid]) {
				dev_cost[nid] = dev_cost[tid] + 1;
				dev_update[nid] = 1;
			}
		}
	}
}

// mark the new explored vertices as the new frontier
__global__ void bfs_update(Node* dev_graph,
	bool* dev_frontier,
	bool* dev_update,
	bool* dev_visited,
	int* dev_edge_list,
	int* dev_cost,
	int* dev_MAX_NODES,
	bool* finish)
{

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < *dev_MAX_NODES && dev_update[tid]) {
		dev_update[tid] = 0;
		dev_frontier[tid] = 1;
		dev_visited[tid] = 1;
		*finish = true;
	}
}

Node graph[MAX_NODES];
bool frontier[MAX_NODES];
bool update[MAX_NODES];
bool visited[MAX_NODES];
int edge_list[MAX_EDGES];
int cost[MAX_NODES];

Node* dev_graph;
bool* dev_frontier;
bool* dev_update;
bool* dev_visited;
int* dev_edge_list;
int* dev_cost;
int* dev_nodes;
bool* dev_finish;


std::vector<int> adj[MAX_NODES];
bool host_vis[MAX_NODES];
int host_cost[MAX_NODES];
int source = 1;


// bfs on cpu
void host_bfs() {
	std::queue<int> Q;
	Q.push(source);
	host_vis[source] = 1;
	while (!Q.empty()) {
		int now = Q.front(); Q.pop();
		int sz = adj[now].size();
		for (int i = 0; i < (int)adj[now].size(); ++i) {
			int child = adj[now][i];
			if (!host_vis[child]) {
				host_vis[child] = 1;
				host_cost[child] = host_cost[now] + 1;
				Q.push(child);
			}
		}
	}
}

void Free()
{
	cudaFree(dev_graph);
	cudaFree(dev_frontier);
	cudaFree(dev_update);
	cudaFree(dev_visited);
	cudaFree(dev_edge_list);
	cudaFree(dev_cost);
	cudaFree(dev_nodes);
	cudaFree(dev_finish);
}

int main()
{


	for (int i = 0; i < MAX_NODES; ++i) {
		graph[i].start = -1;
		frontier[i] = update[i] = visited[i] = host_cost[i] = 0;
	}

	cost[source] = 0;
	frontier[source] = true;
	visited[source] = true;


	FILE* fp = fopen("sample1.txt", "r");
	if (!fp) {
		printf("CANT OPEN THE FILE!!");
	}

	int x, y;
	int i = 0;
	// reading the file, it may take up to 1 min
	while (fscanf(fp, "%d %d", &x, &y) != EOF) {
		if (i == 0) {
			no_of_nodes = x;
			no_of_edges = y;
			i++;
			continue;
		}
		if (graph[x].start == -1)
			graph[x].start = i;
		graph[x].no_of_edges++;
		edge_list[i] = y;
		i++;
		adj[x].push_back(y);
	}



	fclose(fp);
	printf("host started!\n");

	clock_t cpu_start, cpu_end;
	float cpu_time = 0;
	cpu_start = clock();

	host_bfs();

	cpu_end = clock();
	cpu_time = 1000.0 *  (cpu_end - cpu_start) / (1.0 * CLOCKS_PER_SEC);

	printf("host ended!, time = %f ms\n", cpu_time);

	// allocate in GPU
	cudaMalloc((void**)&dev_graph, MAX_NODES * sizeof(Node));
	cudaMalloc((void**)&dev_frontier, MAX_NODES * sizeof(bool));
	cudaMalloc((void**)&dev_update, MAX_NODES * sizeof(bool));
	cudaMalloc((void**)&dev_visited, MAX_NODES * sizeof(bool));
	cudaMalloc((void**)&dev_edge_list, MAX_EDGES * sizeof(int));
	cudaMalloc((void**)&dev_cost, MAX_NODES * sizeof(int));
	cudaMalloc((void**)&dev_nodes, sizeof(int));

	// copying to GPU
	cudaMemcpy(dev_graph, graph, MAX_NODES * sizeof(Node), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_frontier, frontier, MAX_NODES * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_update, update, MAX_NODES * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_visited, visited, MAX_NODES * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cost, cost, MAX_NODES * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_edge_list, edge_list, MAX_EDGES * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_nodes, &MAX_NODES, sizeof(int), cudaMemcpyHostToDevice);

	dim3 block((int)ceil((no_of_nodes) / (1.0 * THREADS_PER_BLOCK)), 1, 1);
	dim3 thread(THREADS_PER_BLOCK, 1, 1);


	cudaMalloc((void**)&dev_finish, sizeof(bool));

	printf("DEVICE STARTED\n");

	cudaEvent_t start, end; float time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	bool stop = true;
	while (stop) {

		stop = false;
		cudaMemcpy(dev_finish, &stop, sizeof(bool), cudaMemcpyHostToDevice);
		bfs << <block, thread >> > (dev_graph, dev_frontier, dev_update, dev_visited, dev_edge_list, dev_cost, dev_nodes);
		cudaDeviceSynchronize();
		bfs_update << <block, thread >> > (dev_graph, dev_frontier, dev_update, dev_visited, dev_edge_list, dev_cost, dev_nodes, dev_finish);
		cudaDeviceSynchronize();
		cudaMemcpy(&stop, dev_finish, sizeof(bool), cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	printf("DEVICE FINISHED, time = %f ms\n", time);
	cudaMemcpy(cost, dev_cost, MAX_NODES * sizeof(int), cudaMemcpyDeviceToHost);
	system("pause");

	Free();

	// testing the results
	for (int i = 0; i < MAX_NODES; ++i) {
		if (cost[i] != host_cost[i]) {
			printf("%d %d for %d\n", cost[i], host_cost[i], i);
			system("pause");
		}
	}
	return 0;
}

