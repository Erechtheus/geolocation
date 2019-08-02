import networkx as nx     # nx can be seemed as an alias of networkx module
import gzip
from representation import parseJsonLine
import datetime
import time

trainingFile="data/train/training.twitter.json.gz" #File with  all ~9 Million training tweets
binaryPath= 'data/binaries/'            #Place to store the results

G = nx.Graph()            # create an empty graph with no nodes and no edges

start = time.time()
with gzip.open(trainingFile,'rb') as file:
    for line in file:
        instance = parseJsonLine(line.decode('utf-8'))

        if instance.userMentions != None and len(instance.userMentions) != 0:
            mentions = instance.userMentions
            userId = instance.userId
            G.add_node(userId)
            for mention in mentions:
                G.add_node(mention)
                if G.has_edge(userId, mention):
                    edge = G.get_edge_data(userId, mention)
                    G.add_edge(userId, mention, frequency=1 +edge['frequency'])

                else:
                    G.add_edge(userId, mention, frequency = 1)

            #Also add co-occuring nodes, mentioned by the user
            for i in range(0,len(mentions)):
                for j in range (i,len(mentions)):
                    nodeA = mentions[i]
                    nodeB = mentions[j]

                    G.add_node(nodeA)
                    G.add_node(nodeB)

                    if G.has_edge(nodeA, nodeB):
                        edge = G.get_edge_data(nodeA, nodeB)
                        G.add_edge(nodeA, nodeB, frequency=1 + edge['frequency'])

                    else:
                        G.add_edge(nodeA, nodeB, frequency=1)

print("Parsing finished after " +str(datetime.timedelta(seconds=round(time.time() - start)))) #Takes approximately 15 minutes
print('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
nx.write_gpickle(G, binaryPath +"graph.gpickle")
nx.write_edgelist(G, binaryPath +"graph.edgelist", data=['frequency'])

#G=nx.read_gpickle("/home/philippe/PycharmProjects/geolocation/data/binaries/graph.gpickle")
