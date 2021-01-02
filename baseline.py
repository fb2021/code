#import socket
import pandas as pd
#from ipwhois import IPWhois
#import struct
#from sklearn.model_selection import StratifiedKFold
import numpy as np
#from sklearn.metrics import confusion_matrix
#from icecream import ic

conf_mat = []

seed=20
np.random.seed(seed)
#import networkx as nx
df=pd.read_csv(r"C:\Users\desktopone\Desktop\labels.irani.csv")
ds=pd.read_csv(r"C:\Users\desktopone\Desktop\domain_ipirani.csv")
#print(df)
#print(ds)
print("*********************************************")

#df=pd.read_csv(r"mb.csv")
#print(df)
df=df.as_matrix()
ds=ds.as_matrix()
#####################################################
#def cidr_to_netmask(cidr):
    #network, net_bits = cidr.split('/')
    #host_bits = 32 - int(net_bits)
    #netmask = socket.inet_ntoa(struct.pack('!I', (1 << 32) - (1 << host_bits)))
    #return network, netmask
#cidr_to_netmask('85.185.64.0/22')

domainList = []
labelsList = []
ip_addresses={}
labels = {}
#asns = {}
#cidrs = {}
#netmasks = {}
for i in df:
    x=i[0].lower()
    #print(x)
    if "orcid.org"==x:
        print("-----------")
        print(i)
        print("-----------")
#    print(i[0])
#    print(i[1])
#    print('----')
    if i[1] == 0:
        labels[x] = 0
        labelsList.append(0)
        domainList.append(x)
    else:
        labels[x] = 1
        labelsList.append(1)
        domainList.append(x)        
for l in ds:
        temp = []
        ip_address_list = l
        if ip_address_list[0].lower() in ip_addresses.keys():
            ip_addresses[ip_address_list[0].lower()].append(ip_address_list[1])
        else:
            temp.append(ip_address_list[1])
            ip_addresses[ip_address_list[0].lower()] = temp
    
        
    
          

        #temp_cidr = []
        #temp_netmask = []
        #asn = {}
        #for ip_address in ip_address_list[2]:
            #ipwhois=IPWhois(ip_address)
            #asn[ip_address] = ipwhois.net.get_asn_dns()[0].to_text().split('|')[0].split('"')[1].strip()
            #cidr = ipwhois.net.get_asn_dns()[0].to_text().split('|')[1]
            #temp_cidr.append(cidr)
            #network, netmask = cidr_to_netmask(cidr)
            #temp_netmask.append(netmask)
        #asns[ip_address_list[0]] = asn
        #cidrs[ip_address_list[0]] = temp_cidr
        #netmasks[ip_address_list[0]] = temp_netmask
    #except:
        #print("You cannot! ",i[0])
#print(labels["vip-lb.wordpress.com"])
#######################################################################
class Node:
    def __init__(self, name=None, name_type=None):
        self.name = name
        self.name_type = name_type



class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}
        # self.connectedToVertex = {}

    # def addConnect(self,nbr, nbr1):
    #     if len(self.connectedToVertex) == 0 or (nbr not in self.connectedToVertex.keys()):
    #         self.connectedToVertex[nbr] = [0, [nbr1]]
    #     else:
    #         temp = self.connectedToVertex[nbr]
    #         temp[0] = temp[0] + 1
    #         temp[1] = temp[1].append(nbr1)
    #         self.connectedToVertex[nbr] = temp
    
    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]



################################################################
class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0
        self.connectedToVertex = {}

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,weight=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)
        # self.vertList[f].addConnect(t, f)
        
        # print(self.connectedToVertex)
        # print(len(self.connectedToVertex))
        # if (t in self.connectedToVertex.keys()):
        #     print(t)
        #     print(f)
        #     print(self.connectedToVertex[t])
        #     print(t)
        #     print(self.connectedToVertex.keys())
        if len(self.connectedToVertex) == 0:
            self.connectedToVertex[t] = [1, [f]]
        elif (t not in self.connectedToVertex.keys()):
            self.connectedToVertex[t] = [1, [f]]
        else:
            # print(t)
            # print(f)
            # print(self.connectedToVertex[t])
            temp = self.connectedToVertex[t]
            # print(temp)
            # print(temp[0])
            # print(temp[1])
            countConnection = temp[0] + 1
            # print('temp[0]' + str(countConnection))
            connection = temp[1]
            # print('temp[1]' + str(temp1))
            connection.append(f)
            self.connectedToVertex[t] = [countConnection, connection]
        # print(self.connectedToVertex)
        

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())











# g = Graph()
# for i in range(6):
#    g.addVertex(i)
# g.addEdge(0,1,5)
# g.addEdge(0,5,2)
# g.addEdge(1,2,4)
# g.addEdge(2,3,9)
# g.addEdge(3,4,7)
# g.addEdge(3,5,3)
# g.addEdge(4,0,1)
# g.addEdge(5,4,8)
# g.addEdge(5,2,1)
# for v in g:
#    for w in v.getConnections():
#        print("( %s , %s )" % (v.getId(), w.getId()))







################################################################################


    
domain_ip_graph = Graph()
        
dict_ip_private={}
dict_ip_public={}
temp=True
ip_dict = {}

for counter in ip_addresses.keys():
#for counter in train_features:
# for iprivate in a:
    iprivate = ip_addresses[counter]
    private = []
    public = []
    for i in iprivate:
        temp = int(i.replace('.',''))
        if ((temp>=127000 and temp<=255000)
            or (temp>=19216800 and temp<=25525500)
            or (temp>=1721600 and temp<=25524000)
            or (temp>=10000 and temp<=255000)):
            if i not in ip_dict.keys():
                ip_dict[i] = Node(i,'private')
            private.append(i)
            domain_ip_graph.addEdge(counter, ip_dict[i].name)
        else:
            if i not in ip_dict.keys():
                ip_dict[i] = Node(i,'public')
            public.append(i)
            domain_ip_graph.addEdge(counter, ip_dict[i].name)
    
    dict_ip_private[iprivate[0]] = []
    dict_ip_public[iprivate[0]] = []        
    if len(private) != 0:
        dict_ip_private[iprivate[0]] = private
    if len(public) != 0:
        dict_ip_public[iprivate[0]] = public
        


    
    
#for i in domain_ip_graph.connectedToVertex.keys():
    #s = i
    #for j in domain_ip_graph.connectedToVertex[i][1]:
       # s = s + " --> " + j
    #print(s)
    
#for v in domain_ip_graph:
     #for w in v.getConnections():
        #print("( %s , %s )" % (v.id, w.id))

        
 ###################################################################       
        
def calculate_weight(a,b): 
    q=set(ip_addresses[a])
    z=set(ip_addresses[b])
    eshterac=q.intersection(z)
    # print(eshterac)
    w = 1- (1/(1+len(eshterac)))
    return w
    #ic()
                
        
        
#G-baseline:        
        
domain_graph_Gbaseline = Graph()
#dict_baseline={}
b_list=[]
item=True

for k in domain_ip_graph.connectedToVertex.keys():
    # print(k)
    # print( domain_ip_graph.connectedToVertex.values())
    # print('------------')
    b_list=domain_ip_graph.connectedToVertex[k]
    # print(k)
    # print(b_list)
    # print('----')
    
    if b_list[0]==1:
#        domain_graph_Gbaseline.addEdge(b_list[1][0],b_list[1][0])
        pass
        
    else:
         for item in range(len(b_list[1])):
            for itemm in range(len(b_list[1])):
                  if item==itemm:
                      pass
                  else:
                      weight = calculate_weight(b_list[1][item],b_list[1][itemm])
                      domain_graph_Gbaseline.addEdge(b_list[1][item],b_list[1][itemm],weight)
#                if item==itemm:
              # domain_graph_Gbaseline.addEdge(b_list[1][item],b_list[1][itemm],1) 
#                elif itemm>item:
#                      weight = calculate_weight(b_list[1][item],b_list[1][itemm])
#                      domain_graph_Gbaseline.addEdge(b_list[1][item],b_list[1][itemm],weight) 


columnNode = []
columnWeight = []
columnDestinationNode = []
for key in domain_graph_Gbaseline.vertList.keys():
    print(key)
    
    for i in domain_graph_Gbaseline.vertList[key].connectedTo.keys():
        columnNode.append(key)
        print(i.id)
        columnWeight.append(domain_graph_Gbaseline.vertList[key].connectedTo[i])
        columnDestinationNode.append(i.id)
        print(domain_graph_Gbaseline.vertList[key].connectedTo[i])
    print('-----')
data = {'Node':columnNode,
        'Weight':columnWeight,
        'Destination_Node':columnDestinationNode}
df = pd.DataFrame (data, columns = ['Node','Weight','Destination_Node'])
df.to_csv(r'G_Base_Line.irani.csv', index = False)


#####################################################################3



























