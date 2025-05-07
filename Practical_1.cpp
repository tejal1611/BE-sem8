#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Graph class representing the adjacency list
class Graph {
    int V;  // Number of vertices
    vector<vector<int>> adj;  // Adjacency list

public:
    Graph(int V) : V(V), adj(V) {}

    // Add an edge to the graph
    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }

    // Parallel Depth-First Search
    void parallelDFS(int startVertex) {
        vector<bool> visited(V, false);
        parallelDFSUtil(startVertex, visited);
    }

    // Parallel DFS utility function
    void parallelDFSUtil(int v, vector<bool>& visited) {
        //#pragma omp critical
       // { 
        visited[v] = true;
        cout << v << " ";
       // }
        #pragma omp parallel for
        for (int i = 0; i < adj[v].size(); ++i) {
            int n = adj[v][i];
            if (!visited[n])
                parallelDFSUtil(n, visited);
        }
    }

    // Parallel Breadth-First Search
    void parallelBFS(int startVertex) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[startVertex] = true;
        q.push(startVertex);

        while (!q.empty()) {
            int v = q.front();
            q.pop();
            cout << v << " ";

            //#pragma omp parallel for
            for (int i = 0; i < adj[v].size(); ++i) {
                int n = adj[v][i];
                if (!visited[n]) {

                    visited[n] = true;
                    q.push(n);
                }
            }
            
        }
    }
};

int main() {
    // Create a graph
    int n;
    cout<<"enter number of node: "<<endl;
    cin>>n;
    Graph g(n);
    int edges;
    cout<<"enter number of edges "<<endl;
    cin>>edges;
    for(int i=0;i<edges;i++){
        int s,e;
        cout<<"enter start and end vertex: "<<endl;
        cin>>s>>e;
        g.addEdge(s,e);
    }
    // Graph g(7);
    // g.addEdge(0, 1);
    // g.addEdge(0, 2);
    // g.addEdge(1, 3);
    // g.addEdge(1, 4);
    // g.addEdge(2, 5);
    // g.addEdge(2, 6);
    
    /*
        0 ---------1
        |         / \
        |        /   \
        |       /     \
        |      /       \
        2 ----3         4
      / |      
     /  |      
    /   |      
    5   6      
    */

    cout << "Depth-First Search (DFS): ";
    double start_time = omp_get_wtime();
    g.parallelDFS(0);
    double end_time = omp_get_wtime();
    cout << endl;
    cout << "Parallel DFS Execution Time: " << end_time - start_time << " seconds" << endl;

    cout << "Breadth-First Search (BFS): ";
    start_time = omp_get_wtime();
    g.parallelBFS(0);
    end_time = omp_get_wtime();
    cout << endl;
    cout << "Parallel BFS Execution Time: " << end_time - start_time << " seconds" << endl;
   
    return 0;
}
