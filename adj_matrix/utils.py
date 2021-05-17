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
#######################
## Caviar parameters ##
#######################


## Selection of networks for deep analysis
nums_cols = {
    "fase_1a": {
        "int": 1,
        "col": "c",
        "pos": (0, 0)
    },
    "fase_1b": {
        "int": 3,
        "col": "c",
        "pos": (0, 1)
    },
    "fase_2": {
        "int": 5,
        "col": "r",
        "pos": (0, 2)
    },
    "fase_3": {
        "int": 6,
        "col": "g",
        "pos": (1, 0)
    },
    "fase_4a": {
        "int": 8,
        "col": "y",
        "pos": (1, 1)
    },
    "fase_4b": {
        "int": 10,
        "col": "y",
        "pos": (1, 2)
    },
}





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



## Create all graphs for each phase
def create_fase_nxs(G, nums, fix, title, color):
    """

    :return:
    """

    pos = {}

    if len(nums) > 1:
        fig, ax = plt.subplots(nums[0], nums[-1], figsize=(17, 8))
        for num in nums:
            pos[num] = nx.drawing.nx_agraph.graphviz_layout(G[num + fix])

            nx.draw(
                G[num + fix],
                pos=pos[num],
                ax=ax[num - 1],
                with_labels=True,
                node_color=color
            )

            ax[num - 1].set_title("Intervención #" + str(num + fix))

    else:
        fig, ax = plt.subplots(nums[0], nums[-1], figsize=(10, 8))
        for num in nums:
            pos[num] = nx.drawing.nx_agraph.graphviz_layout(G[num + fix])

            nx.draw(
                G[num + fix],
                pos=pos[num],
                with_labels=True,
                node_color="g"
            )
            ax.set_title("Intervención #" + str(num + fix))

    fig.suptitle(title, fontsize=20)

    plt.show()



## Networks deep analysis (visual)
def networks_deep_analysis(G, nums_cols, a_type):
    """

    :return:
    """

    ## Initial parameters
    fig, ax = plt.subplots(2, 3, figsize=(20, 15))

    ## Drawing networks
    for sel in nums_cols:
        num = nums_cols[sel]["int"]
        pos = nums_cols[sel]["pos"]

        #### Analysis type 1: visual analysis
        if a_type == "visual":
            col = nums_cols[sel]["col"]
            nx.draw(
                G[num],
                pos=nx.drawing.nx_agraph.graphviz_layout(G[num]),
                ax=ax[pos[0], pos[1]],
                with_labels=True,
                node_color=col
            )

            fig.suptitle("Inspección visual de las redes", fontsize=20)


        #### Analysis type 2: centrality metrics
        else:
            #### Calculating values according to centrality metric
            if a_type == "grado":
                values = [nx.degree_centrality(G[num])[val] for val in nx.degree_centrality(G[num])]
                fig.suptitle("Medida de centralidad (" + a_type + ") en tiempos seleccionados", fontsize=20)
            elif a_type == "intermediación":
                values = [nx.betweenness_centrality(G[num])[val] for val in nx.betweenness_centrality(G[num])]
                fig.suptitle("Medida de centralidad (" + a_type + ") en tiempos seleccionados", fontsize=20)
            elif a_type == "eigenvector":
                values = [nx.eigenvector_centrality(G[num])[val] for val in nx.eigenvector_centrality(G[num])]
                fig.suptitle("Medida de centralidad (" + a_type + ") en tiempos seleccionados", fontsize=20)
            elif a_type == "comunidades":
                fig.suptitle("Detección de comunidades", fontsize=20)
                ###### Initial values
                comms = nx.algorithms.community.greedy_modularity_communities(G[num])
                dict_x2 = {}
                values = []
                val = 0.25
                ###### Community loops
                for i in range(1, len(comms) + 1):
                    node_val = {i: val for i in comms[i - 1]}
                    dict_x2.update(node_val)
                    val += 0.25
                for node in G[num]:
                    values.append(dict_x2[node])
            else:
                raise NameError("Análisis no válido")

            #### Creating graph
            nx.draw(
                G[num],
                pos=nx.drawing.nx_agraph.graphviz_layout(G[num]),
                ax=ax[pos[0], pos[1]],
                with_labels=True,
                cmap=plt.get_cmap('inferno'),
                node_color=values
            )


        ax[pos[0], pos[1]].set_title("Intervención #" + str(num) + " (" + sel + ")")

    plt.show()



## Dataframes with numeric calculations - each intervention
def df_centrality_metrics_periods(G, a_type):
    """

    :return:
    """

    ##
    dict_x = {}

    for num in G:

        if a_type == "eigenvector":
            dfx = pd.DataFrame.from_dict(nx.eigenvector_centrality(G[num]), orient="index")
        elif a_type == "intermediación":
            dfx = pd.DataFrame.from_dict(nx.betweenness_centrality(G[num], normalized=True), orient="index")

        dfx.sort_values(by=0, ascending=False, inplace=True)
        dfx.reset_index(inplace=True)
        dfx[num] = dfx.apply(lambda x: (x["index"], round(x[0], 2)), axis = 1)
        dfx = dfx.loc[0:2, [num]]
        dict_x[num] = dfx

    ##
    dfx = pd.concat([dict_x[num] for num in dict_x], axis=1)
    dfx.index = range(1, 4)
    print("Calculo de métrica de centralidad (" + a_type + ") para todas las intervenciones" )
    display(dfx)



## Dataframes with numeric calculations - mean
def df_centrality_metrics_mean(G, a_type):
    """

    :return:
    """

    ##
    dict_x = {}

    for num in G:

        if a_type == "eigenvector":
            dfx = pd.DataFrame.from_dict(nx.eigenvector_centrality(G[num]), orient="index")
        elif a_type == "intermediación":
            dfx = pd.DataFrame.from_dict(nx.betweenness_centrality(G[num], normalized=True), orient="index")

        dfx.columns = [num]
        dfx[num] = dfx[num].apply(lambda x: round(x, 2))
        dict_x[num] = dfx

    ##
    dfx = pd.concat([dict_x[num] for num in dict_x], axis=1)
    dfx[dfx.isna()] = 0
    dfx["mean"] = round(dfx.mean(axis=1), 2)
    dfx.sort_values(by="mean", ascending=False, inplace=True)
    dfx = dfx.loc[:, ["mean"]].iloc[0:5, :]
    print("Promedio de métrica de centralidad (" + a_type + ") para todas las intervenciones" )
    display(dfx)




"----------------------------------------------------------------------------------------------------------------------"
## END OF FILE ##
"----------------------------------------------------------------------------------------------------------------------"