from my_module import tools


n_objects_row = 150
n_objects_col = 200
n_clusters_row = 2
n_clusters_col = 2
theta = [[0.3, 0.5], [0.7, 0.1]]

tools.generate_bipartite_graph(n_objects_row, n_objects_col, n_clusters_row, n_clusters_col, theta)
tools.visualize_matrix()