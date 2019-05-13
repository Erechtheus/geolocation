import networkx as nx     # nx can be seemed as an alias of networkx module

import gzip
from representation import parseJsonLine, Place, extractPreprocessUrl
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
            G.add_node(instance.userId)
            for mention in mentions:
                G.add_node(mention)
                if G.has_edge(instance.userId, mention):
                    edge = G.get_edge_data(instance.userId, mention)
                    G.add_edge(instance.userId, mention, frequency=1 +edge['frequency'])

                else:
                    G.add_edge(instance.userId, mention, frequency = 1)
print("Parsing finished after " +str(datetime.timedelta(seconds=round(time.time() - start)))) #Takes approximately 15 minutes
nx.write_gpickle(G, binaryPath +"graph.gpickle")
nx.write_edgelist(G, binaryPath +"graph.edgelist", data=['frequency'])

#G=nx.read_gpickle("test.gpickle")