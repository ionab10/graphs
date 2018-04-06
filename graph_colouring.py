import pandas as pd
import networkx as nx
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from operator import or_, and_
from functools import reduce
import json
import numpy as np


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif type(obj)==set:
            return list(obj)
        else:
            return super(MyEncoder, self).default(obj)

def exceeds_max(G, day, maximum):
    if maximum:
        classes_for_day=[n for n in G.nodes() if G.node[n]['col']==day]
        try:
            return len(reduce(or_, [G.node[c]['students'] for c in classes_for_day])) > maximum
        except:
            return False
    else:
        return False
        
def get_available_colour(G, n, all_cols, allow_conflicts=False, max_students=None):
    unavailable={G.node[n]['col'] for n in G.neighbors(n)}
    available=sorted(set(all_cols) - unavailable)
    for day in available:
        if not exceeds_max(G, day, max_students):
            return day

    if allow_conflicts:
        G.node[n]['clash']=True
        clashes=sorted(clashes_per_day(G).items(), key=lambda x: x[1])
        for day,n_clashes in clashes:
            if not exceeds_max(G, day, max_students):
                for nb in G.neighbors(n):
                    if G.node[nb]['col']==day:
                        G.node[nb]['clash']=True
                return day

        raise ValueError('Too many students')
    else:
        raise ValueError('Too many students or not enough days')
        
def clashes_per_day(G):
    n_clashes={}
    for n in G.nodes():
        if G.node[n]['col']==-1:
            continue
        elif G.node[n]['clash']:
            try: 
                n_clashes[G.node[n]['col']]+=G.node[n]['size']
            except:
                n_clashes[G.node[n]['col']]=G.node[n]['size']
        else:
            try: 
                n_clashes[G.node[n]['col']]+=0
            except:
                n_clashes[G.node[n]['col']]=0
    return n_clashes

def try_coloring(G, num_colours=None, allow_conflicts=False, order='size', max_students=None):
    #set all nodes to -1
    for n in G.nodes():
        G.node[n]['col']=-1
        G.node[n]['clash']=False
        
    #available colours/days
    if num_colours:
        colours=list(range(num_colours))
    else:
        colours=list(range(len(G.nodes())))
    
    #for each node colour with the first available colour
    for n,size in sorted(nx.get_node_attributes(G,order).items(), key=lambda x: x[1], reverse=True):
        if G.node[n]['col']==-1:
            G.node[n]['col']=get_available_colour(G, n, colours, allow_conflicts=allow_conflicts, max_students=max_students)

    return G

#read in data as table
df=pd.read_csv('data.csv')

#get list of subjects
subjects=list(set(df['Subj']))

# create dictionary of students in each subject (ex. {PHY: {1234, 1235, 1236}})
class_dict={}
for subject in subjects:
    class_dict[subject]=set(df.query("Subj == '{}'".format(subject))['No.'])

#Create graph
G=nx.Graph()
G.add_nodes_from(subjects)  #vertices are subjects

#for each pair of classes, check if there is a student in both
for x in combinations(subjects, 2):
    class1=x[0]
    class2=x[1]
    # if the intersection of the set of students in class1 and the set of students in class2 is not empty 
    #(ie there is at least one student in both classes)...
    if len(class_dict[class1] & class_dict[class2])>0:
        G.add_edge(class1,class2)    #edge if two classes has a common student

for n in G.nodes():
    G.node[n]['size']=len(class_dict[n])
    G.node[n]['students']=class_dict[n]

#colouring with no restriction on number of days
#print("Non-restrictive colouring: ",nx.coloring.greedy_color(G))

num_colours=3
max_students=3

G=try_coloring(G, num_colours=num_colours, allow_conflicts=True, max_students=max_students)
cols=nx.get_node_attributes(G,'col')

with open('schedule.txt','w') as f:
    for i in range(num_colours):
        f.write('Day {}: '.format(i) + ', '.join([x[0] for x in cols.items() if x[1]==i]) + '\n')
        
class_df=pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

with open('clashes.txt','w') as f:
    
    for day in set(class_df.col):
        temp=class_df.query("col=={} & clash==True".format(day))
        try:
            clashing_students=reduce(and_, temp.students)
        except:
            clashing_students={}
        f.write('Day {}: '.format(day) + str(list(temp.index)) + '\t' + ', '.join([str(x) for x in clashing_students]) + '\n')
            
with open('G.json', 'w') as f:
    json.dump(dict(G.nodes(data=True)), f, cls=MyEncoder, indent=4, sort_keys=True)

#draw graph (different colour for different days)
nx.draw(G, with_labels=True, node_color=[cols[n] for n in G.nodes()], cmap=plt.cm.Blues)