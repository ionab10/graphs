import pandas as pd
import networkx as nx
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from operator import or_, and_
from functools import reduce
import json
import numpy as np


# JSON encoder because json.dump doesn't like numpy or sets
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

# find elements that appear in at least two sets out of a list of sets
# probably exists a function already but i was too lazy to find it
def pairwise_and(sets):
    temp=set()
    duplicates=set()
    for s in sets:
        for x in s:
            if x not in temp:
                temp.add(x)
            else:
                duplicates.add(x)
    return duplicates


# return True if adding the class n to day would mean for more than 'maximum' students on that day
# Client wanted a bound on the number of students scheduled each day
def exceeds_max(G, day, n, maximum):
    if maximum:
        classes_for_day=[node for node in G.nodes() if G.node[node]['col']==day]
        try:
            existing_students=[G.node[c]['students'] for c in classes_for_day]
            n_students=len(reduce(or_, existing_students + [G.node[n]['students']]))
            return n_students > maximum
        except:
            return False
    else:
        return False
        
#try to find an availaby colour/day
#first check for non-conflicting colours
#if conflicts are allowed minimize the conflict
#if you can't add students to any days without maxing out the max_students limit then exit with ValueError
def get_available_colour(G, n, all_cols, allow_conflicts=False, max_students=None):
    unavailable={G.node[n]['col'] for n in G.neighbors(n)}  #colours of neighbouring vertices are unavailable
    available=sorted(set(all_cols) - unavailable)
    for day in available:
        if not exceeds_max(G, day, n, max_students):     #try to add class n
            return day

    if allow_conflicts:               #if we allow some conflicts
        G.node[n]['clash']=True
        clashes=sorted(clashes_per_day(G).items(), key=lambda x: x[1])   #least conflicts
        for day,n_clashes in clashes:
            if not exceeds_max(G, day, n, max_students):
                for nb in G.neighbors(n):
                    if G.node[nb]['col']==day:
                        G.node[nb]['clash']=True
                return day

        raise ValueError('Too many students')
    else:
        raise ValueError('Too many students or not enough days')
        
#calculates the number of students with conflicts per day
def clashes_per_day(G):
    n_clashes={}
    for n in G.nodes():
        if G.node[n]['col']==-1:
            continue
        elif G.node[n]['clash']:
            try: 
                n_clashes[G.node[n]['col']]=n_clashes[G.node[n]['col']].union(G.node[n]['students'])
            except:
                n_clashes[G.node[n]['col']]=G.node[n]['students']
        elif G.node[n]['col'] not in n_clashes.keys():
            n_clashes[G.node[n]['col']]=set()   
        
    n_clashes=dict([(k,len(v)) for k,v in n_clashes.items()])
    return n_clashes

#main colouring algorithm
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

#manual constraints
num_colours=3
max_students=3

#run algorithm
G=try_coloring(G, num_colours=num_colours, allow_conflicts=True, max_students=max_students)
cols=nx.get_node_attributes(G,'col')


#outputs

#schedule (classes for each day)
with open('schedule.txt','w') as f:
    for i in range(num_colours):
        f.write('Day {}: '.format(i) + ', '.join([x[0] for x in cols.items() if x[1]==i]) + '\n')
        
class_df=pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

#classes with conflicts and students with conflicts on each day
with open('clashes.txt','w') as f:
    
    for day in set(class_df.col):
        temp=class_df.query("col=={} & clash==True".format(day))
        clashing_students=pairwise_and(list(temp.students))
        f.write('Day {}: '.format(day) + str(list(temp.index)) + '\t' + ', '.join([str(x) for x in clashing_students]) + '\n')

#all the graph data as a JSON
with open('G.json', 'w') as f:
    json.dump(dict(G.nodes(data=True)), f, cls=MyEncoder, indent=4, sort_keys=True)

#draw graph (different colour for different days)
nx.draw(G, with_labels=True, node_color=[cols[n] for n in G.nodes()], cmap=plt.cm.Blues)
