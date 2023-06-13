import sys

'''
WRITE YOUR CODE BELOW.
'''
import numpy as np
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function
    
    #Add to Network
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")
    
    #Add Edges (Parent, Child)
    BayesNet.add_edge("temperature","faulty gauge")
    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("gauge","alarm")
    BayesNet.add_edge("faulty alarm","alarm")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    
    #Probabilities for Temp
    cpd_temperature = TabularCPD('temperature', 2, values=[[0.8], [0.2]])
    
    #Probabilities for Faulty Alarm
    cpd_faulty_alarm = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])
    
    #Probabilities for Faulty Gauge
    cpd_faulty_gauge = TabularCPD('faulty gauge', 2, values=[[0.95, 0.2], \
                    [ 0.05, 0.8]], evidence=['temperature'], evidence_card=[2])

    
    #Probabilities for Gauge
    cpd_gauge = TabularCPD('gauge', 2, values=[[0.95, 0.05, 0.2, 0.8], \
                    [0.05, 0.95, 0.8, 0.2]], evidence=['faulty gauge', 'temperature'], evidence_card=[2, 2])

    #Probabilities for Alarm
    cpd_alarm = TabularCPD('alarm', 2, values=[[0.9, 0.1, 0.55, 0.45], \
                    [0.1, 0.9, 0.45, 0.55]], evidence=['faulty alarm', 'gauge'], evidence_card=[2, 2])
    
   
    
    bayes_net.add_cpds(cpd_temperature, cpd_faulty_alarm, cpd_faulty_gauge, cpd_gauge, cpd_alarm)

    
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system. (T/F)
    """
    
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values

    
    return alarm_prob[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""

    
    
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values
    
 
    return gauge_prob[1]


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],evidence={'alarm':1,'faulty gauge':0, 'faulty alarm' :0}, joint=False)
    temp_prob = conditional_prob['temperature'].values

    
    
    return temp_prob[1]


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()

    #Add to Network
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")
    
    #Add Edges (Parent, Child)
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("C","CvA")
    BayesNet.add_edge("A","CvA")

    #Probabilities for A
    cpd_A = TabularCPD('A', 4, values=[[.15], [.45], [.30], [.10]])

    #Probabilities for B
    cpd_B = TabularCPD('B', 4, values=[[.15], [.45], [.30], [.10]])
    
    #Probabilities for C
    cpd_C = TabularCPD('C', 4, values=[[.15], [.45], [.30], [.10]])
    
    
    #Probabilities for AvB
    cpd_AvB=TabularCPD("AvB",3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1],\
                       [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                       [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],\
                       evidence=["A","B"], evidence_card=[4, 4])
    
    
    
    cpd_BvC=TabularCPD("BvC",3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1],\
                       [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                       [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],\
                       evidence=["B","C"], evidence_card=[4, 4])
    
    cpd_CvA=TabularCPD("CvA",3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1],\
                       [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                       [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],\
                       evidence=["C","A"], evidence_card=[4, 4])
    
    
    BayesNet.add_cpds(cpd_A,cpd_B,cpd_C,cpd_AvB,cpd_BvC,cpd_CvA) 
    
    #All Matches and Teams have same distro - only need to do once
    m = cpd_AvB.values
    print(m)
    print(m[0][0][1])
    t = cpd_A.values
    print(t)
    
    
    
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    
    solver = VariableElimination(bayes_net)
    conditional_prob_BvC = solver.query(variables=['BvC'],evidence={'AvB':0,'CvA':2}, joint=False)
    posterior = conditional_prob_BvC['BvC'].values

    
    return posterior 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """  
    
    
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    
    #All Matches and Teams have same distro - only need to do once
    m = AvB_cpd.values
    print(m)
    t = A_cpd.values
    print(t)

    
    #Rand generate  initial state (AvB and CvA already set)
    sample = list(initial_state)    
    
    if len(sample) == 0:
        A=np.random.randint(4)
        B=np.random.randint(4)
        C=np.random.randint(4)
        AvB=0
        BvC=np.random.randint(3)
        CvA=2
        sample=[A,B,C,AvB,BvC,CvA]
        
        
    sampleIndex=np.array([0,1,2,4]) #Sampling only changable variables
    index=np.random.choice(sampleIndex)
    
    #Getting Sample value for A
    if index==0:
        
        distA = [0] * 4
        
     
        distA[0]=m[0][0][sample[1]]*m[2][sample[2]][0]*t[0]
        
        distA[1]=m[0][1][sample[1]]*m[2][sample[2]][1]*t[1]
        
        distA[2]=m[0][2][sample[1]]*m[2][sample[2]][2]*t[2]
        
        distA[3]=m[0][3][sample[1]]*m[2][sample[2]][3]*t[3]
        
        total=sum(distA)
        
        distA = [distA[0]/total , distA[1]/total, distA[2]/total, distA[3]/total]
        
        #Updating A Value with sample value
        sample[0]=np.random.choice(4,p=distA)    
    
    #Getting Sample value for B
    elif index==1:
        
        distB =[0] * 4
       
        distB[0]=m[0][sample[0]][0]*m[sample[4]][0][sample[2]]*t[0]
        distB[1]=m[0][sample[0]][1]*m[sample[4]][1][sample[2]]*t[1]
        distB[2]=m[0][sample[0]][2]*m[sample[4]][2][sample[2]]*t[2]
        distB[3]=m[0][sample[0]][3]*m[sample[4]][3][sample[2]]*t[3]        
        
        
        
        total=sum(distB)
        
        distB = [distB[0]/total , distB[1]/total, distB[2]/total, distB[3]/total]
        
        #Updating B Value with sample value
        sample[1]=np.random.choice(4,p=distB)
    
    
    
    #Getting Sample value for C
    elif index==2:
        distC= [0] * 4
        
        distC[0]=m[sample[4]][sample[1]][0]*m[2][0][sample[0]]*t[0]
        distC[1]=m[sample[4]][sample[1]][1]*m[2][1][sample[0]]*t[1]
        distC[2]=m[sample[4]][sample[1]][2]*m[2][2][sample[0]]*t[2]
        distC[3]=m[sample[4]][sample[1]][3]*m[2][3][sample[0]]*t[3]
     

        
        total=sum(distC)
        
        distC = [distC[0]/total, distC[1]/total, distC[2]/total, distC[3]/total]
        
        sample[2]=np.random.choice(4,p=distC)
    
    #Getting Sample value for BvC
    else:
        distBvC = [m[0][sample[1]][sample[2]],m[1][sample[1]][sample[2]],m[2][sample[1]][sample[2]]]
        sample[index]=np.random.choice(3,p=distBvC)
 
    return tuple(sample)


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    m = AvB_cpd.values
    t = A_cpd.values
    sample = tuple(initial_state)   
    
    #If sample is empty
    if len(sample) == 0:
        A=np.random.randint(4)
        B=np.random.randint(4)
        C=np.random.randint(4)
        AvB=0
        BvC=np.random.randint(3)
        CvA=2
        sample=[A,B,C,AvB,BvC,CvA]
    #Generating initial prob
    prob=t[sample[0]]*t[sample[1]]*t[sample[2]]*\
         m[0][sample[0]][sample[1]]*\
         m[sample[4]][sample[1]][sample[2]]*\
         m[2][sample[2]][sample[0]]
    
    #Generating Jump Step 
    A=np.random.randint(4)
    B=np.random.randint(4)
    C=np.random.randint(4)
    AvB=0
    BvC=np.random.randint(3)
    CvA=2
    Jsam=[A,B,C,AvB,BvC,CvA]
    
    #Generating Jump Prob
    Jprob=t[Jsam[0]]*t[Jsam[1]]*t[Jsam[2]]*\
         m[0][Jsam[0]][Jsam[1]]*\
         m[Jsam[4]][Jsam[1]][Jsam[2]]*\
         m[2][Jsam[2]][Jsam[0]]
    
    
    #Compare Jump Step to uniform & return value
    r=min(1,Jprob/prob)
    u=np.random.uniform(0,1)
    if u<r:
        sample=tuple(Jsam)
    else:
        sample=tuple(sample)
    
    
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = np.array([0,0,0]) # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = np.array([0,0,0]) # posterior distribution of the BvC match as produced by MH
    
    delta=0.001
    N=1000
    
    
    #gibbs sampling
    BvC = np.array([0,0,0])
    NGibbs=0
    sampleGibbs=Gibbs_sampler(bayes_net, initial_state)
    while NGibbs < N:
        sampleGibbs=Gibbs_sampler(bayes_net, sampleGibbs)
        Gibbs_count+=1
        if sampleGibbs[4]==0:
            BvC[0] = BvC[0] + 1
        elif sampleGibbs[4]==1:
            BvC[1] = BvC[1] + 1
        else:
            BvC[2] = BvC[2] + 1
        diffGibbs =abs((BvC/Gibbs_count) - Gibbs_convergence)
        
        if diffGibbs[0]<=delta and diffGibbs[1]<=delta and diffGibbs[2]<=delta:
            NGibbs+=1
        else:
            NGibbs=0
        Gibbs_convergence=BvC/Gibbs_count
        
        #Break if iterations exceeds max iterations (N)
        if NGibbs>=N:
            break
            
    Gibbs_convergence=list(Gibbs_convergence)
    
    #MH sampling
    BvC = np.array([0,0,0])
    N_MH = 0
    pastsamp =MH_sampler(bayes_net, initial_state)
    while N_MH < N:
        sampleMH = MH_sampler(bayes_net,pastsamp)
        if sampleMH == pastsamp:
            MH_rejection_count+=1
        else:
            MH_count+=1
            if sampleMH[4]==0:
                BvC[0] = BvC[0] + 1
            elif sampleMH[4]==1:
                BvC[1] = BvC[1] + 1
            else:
                BvC[2] = BvC[2] + 1
        diffMH=abs((BvC/MH_count) - MH_convergence)
        
        if diffMH[0]<=delta and diffMH[1]<=delta and diffMH[2]<=delta:
            N_MH+=1
        else:
            N_MH=0
        MH_convergence = BvC/MH_count
        
        pastsamp=sampleMH
        
        if N_MH>=N:
            break
            
    MH_convergence=list(MH_convergence)
    

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 10
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return  "Chase McGrail"
