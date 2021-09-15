# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        import math 

        for i in range(self.iterations):
            states = self.mdp.getStates()
            counter = util.Counter()
            for state in states:
                bestVal = -math.inf
                possibleAvtions = self.mdp.getPossibleActions(state)
                for action in possibleAvtions:
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > bestVal:
                        bestVal = qVal
                if bestVal == -math.inf:
                    counter[state] = 0
                else:            
                    counter[state] = bestVal
            self.values = counter
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        stateProbabilitySet = self.mdp.getTransitionStatesAndProbs(state, action)
        qval = 0
        for stateProbability in stateProbabilitySet:
            nextState = stateProbability[0]
            probability = stateProbability[1]
            qval += probability * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        import math
        bestPolicy = None
        bestValue = -math.inf
        possibleActions = self.mdp.getPossibleActions(state)
        for action in possibleActions:
            qVal = self.computeQValueFromValues(state, action)
            if qVal > bestValue:
                bestValue = qVal
                bestPolicy = action
        return bestPolicy



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        import math

        states = self.mdp.getStates()
        for i in range(self.iterations):
            updateStateIndex = i % len(states)
            updateState = states[updateStateIndex]
            possibleActions = self.mdp.getPossibleActions(updateState)
            if len(possibleActions) == 0:
                self.values[updateState] = 0
            elif self.mdp.isTerminal(updateState):
                continue
            else:
                maxQVal = -math.inf
                for action in possibleActions:
                    qVal = self.computeQValueFromValues(updateState, action)
                    if qVal > maxQVal:
                        maxQVal = qVal                                
                self.values[updateState] = maxQVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        from util import PriorityQueue
        from math import inf

        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            else:
                possibleActions = self.mdp.getPossibleActions(state)
                for action in possibleActions:
                    probStateSet = self.mdp.getTransitionStatesAndProbs(state, action)
                    for probState in probStateSet:
                        nextState = probState[0]
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState] = {state} 
        
        PQ = PriorityQueue()

        for state in states:
            if not self.mdp.isTerminal(state):
                currentStateValue = self.values[state]
                possibleActions = self.mdp.getPossibleActions(state)
                maxQVal = -inf
                for action in possibleActions:
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > maxQVal:
                        maxQVal = qVal
                difference = abs(currentStateValue - maxQVal)
                PQ.push(state, -difference)


        for i in range(self.iterations):
            if PQ.isEmpty():
                break
            state = PQ.pop()
            if not self.mdp.isTerminal(state):
                maxQVal = -inf
                possibleActions = self.mdp.getPossibleActions(state)
                for action in possibleActions:
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > maxQVal:
                        maxQVal = qVal
                self.values[state] = maxQVal
            statePredecessors = predecessors[state]
            for p in statePredecessors:
                pValue = self.values[p]
                pPossibleActions = self.mdp.getPossibleActions(p)
                maxPQVal = -inf
                for action in pPossibleActions:
                    pqVal =  self.computeQValueFromValues(p, action)
                    if pqVal > maxPQVal:
                        maxPQVal = pqVal        
                pDifference = abs(pValue - maxPQVal)      
                if pDifference > self.theta:
                    PQ.update(p, -pDifference)  





