import numpy as np
from math import *

def compute_purity_entropy(coherent_array):
    clusters = np.transpose(coherent_array)
    clusters_none_zero = []
    entropy, purity = [], []
    for cluster in clusters:
        cluster = np.array(cluster)
        if cluster.sum() == 0.0:
            continue
        clusters_none_zero.append(cluster)
        cluster = cluster / float(cluster.sum())
        e = (cluster * [log((x+1e-4), 2) for x in cluster]).sum()
        p = cluster.max()
        entropy += [e]
        purity	+= [p]

    counts = np.array([c.sum() for c in clusters_none_zero])
    coeffs = counts / float(counts.sum())
    entropy = -(coeffs * entropy).sum()
    purity = (coeffs * purity).sum()
    return entropy, purity

def get_f_score(array):
    final_score = .0
    support_list = []
    fscore_list = []
    index_list = []
    for n in range(0, array.shape[1]):
        index = np.argmax(array[:,n])
        precision = float(np.max(array[:,n],axis=0))/float(np.sum(array[index]))
        if np.sum(array[:,n]) == 0:
            continue
        recall = float(np.max(array[:,n],axis=0))/float(np.sum(array[:,n]))
        fscore = (2*recall*precision)/(recall+precision)
        fscore_list.append(fscore)
        support_list.append(array[index, n])
        index_list.append(index)
        
    TP_list = []
    fscore_list = []
    
    for n in range(0, 4):
        TP = .0
        precision_all = .0
        if n not in index_list:
            continue
        for m in range(0,len(index_list)):
            if index_list[m] == n:
                TP+= array[n,m]
                precision_all += np.sum(array[:,m])
        precision =  TP/precision_all
        recall = TP/np.sum(array[n])
        fscore = (2*recall*precision)/(recall+precision)
        
        TP_list.append(TP)
        fscore_list.append(fscore)
    
    TP_list = np.asarray(TP_list)
    for n in range(0,len(fscore_list)):
        final_score += fscore_list[n] * (TP_list[n]/np.sum(TP_list))
    return final_score