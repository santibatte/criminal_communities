#!/usr/bin/env python3



import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt





def create_adj_matrix(df):
        #formato adecuado a las columnas de interes
    for col in ['CrimeIdentifier', 'OffenderIdentifier']:
        df[col] = df[col].astype('category')
    for col in ['CrimeIdentifier', 'OffenderIdentifier']:
        df[col] = df[col].astype('int64')

    #modifico el numero de crimen y de criminal para que sean numeros consecutivos
    #este paso es importante para reducir el tamaño de la matriz a justo lo que necestiamos
    df['Crimen'] = pd.factorize(df['CrimeIdentifier'])[0]
    df['Criminales'] = pd.factorize(df['OffenderIdentifier'])[0]

    #convierto a np array (para hacer más rapido la matriz rala)
    column=df['Crimen']
    column=np.array(column)
    row=df['Criminales']
    row=np.array(row)

    #agrego una columna de unos para indicar el match
    valor=np.ones((len(df),), dtype=int)
    f=np.column_stack([valor,row,column])

    #datos como matriz rala
    crime_matrix = csr_matrix((f[:,0], (f[:,1], f[:,2])), shape=(row.max() + 1, f[:,2].max() + 1))

    #matriz de adjacencia
    A=crime_matrix*crime_matrix.T

    return A


def create_graph_object(A):
    """
    A is a sparse adjacence matrix
    G is a graph object
    """


    #creo objeto grafo
    # ceros en la diagonal (no self loops)
    A.setdiag(0)
    # me quedo solo con los cruces donde hubo crimenes en comun
    A.eliminate_zeros()
    # creo grafo
    G = nx.from_scipy_sparse_matrix(A)
    # elimino los nodos sin ningun edge
    G.remove_nodes_from(list(nx.isolates(G)))

    Gr_top = G.subgraph(G)


    return G

def filter_number_connections(number_concetions, A, G):
    G_2 = nx.from_scipy_sparse_matrix(A)
    G_2.remove_nodes_from(list(nx.isolates(G)))

    #este loop recorta todos los nodos con menos edges que el numero que determinemos
    G_2_edge_list = list( G_2.edges() )
    G_2_edge_weight_dict = nx.get_edge_attributes(G_2, 'weight' )
    for e in G_2_edge_list:
        if G_2_edge_weight_dict[e] < number_concetions: #aqui eliges el punto de corte (1 o 2 o 5 o 100 dependiendo el problema)
            G_2.remove_edge(e[0],e[1])
    G_2.remove_nodes_from(list(nx.isolates(G_2)))
    G_2.number_of_nodes()

    return G_2



def summary_analysis(df, gr):
    a=list(gr.nodes)
    df_filtered = df[df['Criminales'].isin(a)]
    crime_type= df_filtered['CrimeType1'].value_counts().head()

    municipality = df_filtered['Municipality'].value_counts().head(10)

    crime_location = df_filtered['CrimeLocation'].value_counts().head()

    gender = df_filtered['OffenderGender'].value_counts()
    n_youth = df_filtered['NumberYouthOffenders'].value_counts()



    print('CRIME TYPE \n', crime_type)
    print('MUNICIPALITY \n', municipality)
    print('\n CRIME LOCATION\n', crime_location)
    print('\n GENDER\n', gender)
    print('\n NUMBER_YOUTH\n', n_youth)


def centrality_analysis(gr_top):
    #""""
    #args:

    #returns eigenvector betweness and centrality for each nodes
    #plts 3 graphs
    #"""

    eigenvector=nx.eigenvector_centrality(gr_top)
    betwneeness=nx.betweenness_centrality(gr_top)
    degree=nx.degree_centrality(gr_top)

    data = list(eigenvector.items())
    H=np.array(data)
    plt.hist(H[:,1])
    plt.title('eigenvector centrality')
    plt.show()

    data = list(betwneeness.items())
    H=np.array(data)
    plt.hist(H[:,1])
    plt.title('betwenness centrality')
    plt.show()

    data = list(degree.items())
    H=np.array(data)
    plt.hist(H[:,1])
    plt.title('degree centrality')
    plt.show()


    return eigenvector,betwneeness,degree





"----------------------------------------------------------------------------------------------------------------------"
######################
## Caviar functions ##
######################


## Function to download caviar data
def download_caviar_data():
    """

    :return:
    """

    phases = {}
    G = {}
    for i in range(1,12):
      var_name = "phase" + str(i)
      file_name = "https://raw.githubusercontent.com/ragini30/Networks-Homework/main/" + var_name + ".csv"
      phases[i] = pd.read_csv(file_name, index_col = ["players"])
      phases[i].columns = "n" + phases[i].columns
      phases[i].index = phases[i].columns
      G[i] = nx.from_pandas_adjacency(phases[i])
      G[i].name = var_name

    return phases, G



## Graph exploring adyacency matrix throughout time
def adys_in_time_plot(G):
    """

    :return:
    """

    ## Dataframe with data to plot
    nodes_evol = pd.DataFrame.from_dict(
        {
            "nodes": [G[i].number_of_nodes() for i in G],
            "edges": [G[i].number_of_edges() for i in G]
        }
    )
    nodes_evol.index = ["int_" + str(num) for num in range(1, 12)]

    ## Creating plot
    fig, ax = plt.subplots(figsize=(13, 7))

    nodes_evol["edges"].plot(kind="line", marker="o", color="b", legend=True)
    nodes_evol["nodes"].plot(kind="bar", legend=True, color="g")

    ax.set_xlabel("Intervenciones a lo largo del tiempo", fontsize=15)
    ax.set_ylabel("Conteo de elementos", fontsize=15)

    plt.xticks(rotation=0)

    fig.suptitle("Exploración de matrices de adyacencia (nodos y aristas)", fontsize=20)

    plt.show()




"----------------------------------------------------------------------------------------------------------------------"
## END OF FILE ##
"----------------------------------------------------------------------------------------------------------------------"