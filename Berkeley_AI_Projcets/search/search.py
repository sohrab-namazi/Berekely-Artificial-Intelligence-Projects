# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    from util import Stack

    expanded = []
    StartLocation = problem.getStartState()
    CurrentState = StartLocation, [], 0
    stack = Stack()
    stack.push(CurrentState)
    
    while not stack.isEmpty():
        state, Path, dist  = stack.pop()
        if problem.isGoalState(state):
            break
        Adjs = problem.getSuccessors(state)
        expanded.append(state)
        for Adj in Adjs:
            nextCandidateState = Adj[0]  
            if not nextCandidateState in expanded:
                nextCandidatePath = Path + [Adj[1]]
                nextCandidateDist = dist + 1 
                stack.push((nextCandidateState, nextCandidatePath, nextCandidateDist)) 

    return Path  

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Queue
    from game import Directions

    expanded = []
    queue = Queue()
    StartLocation = problem.getStartState()
    StartState = StartLocation, [], 0
    queue.push(StartState)

    while not queue.isEmpty():
        State, Path, dist  = queue.pop()
        if (State in expanded):
            continue
        expanded.append(State)
        if problem.isGoalState(State):
            break
        Adjs = problem.getSuccessors(State)
        for Adj in Adjs:
            nextCandidateState = Adj[0]
            if not nextCandidateState in expanded:
                nextCandidateDist = dist + 1
                nextCandidatePath = Path + [Adj[1]]
                queue.push((nextCandidateState, nextCandidatePath, nextCandidateDist))
    

    return Path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    from game import Directions

    expanded = []
    pq = PriorityQueue()
    startLocation = problem.getStartState()
    startState = startLocation, [], 0
    pq.push(startState, 0)

    while not pq.isEmpty():
        expandingLocation, expandingPath, expandingDist = pq.pop()
        if (expandingLocation in expanded):
            continue
        expanded.append(expandingLocation)
        if (problem.isGoalState(expandingLocation)):
            break
        adjs = problem.getSuccessors(expandingLocation)
        for adj in adjs:
            if (not adj[0] in expanded):
                nextLocation = adj[0]
                nextDist = expandingDist + adj[2] 
                nextPath = expandingPath + [adj[1]]
                pq.push((nextLocation, nextPath, nextDist), nextDist)

    return expandingPath      

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    from game import Directions

    expanded = []
    pq = PriorityQueue()
    startLocation = problem.getStartState()
    startState = startLocation, [], 0
    pq.push(startState, 0)

    while not pq.isEmpty():
        expandingLocation, expandingPath, expandingDist = pq.pop()
        if (expandingLocation in expanded):
            continue
        expanded.append(expandingLocation)
        if (problem.isGoalState(expandingLocation)):
            break
        adjs = problem.getSuccessors(expandingLocation)
        for adj in adjs:
           if (not adj[0] in expanded):
                nextLocation = adj[0]
                nextDist = expandingDist + adj[2] 
                nextPath = expandingPath + [adj[1]]
                pq.push((nextLocation, nextPath, nextDist), nextDist + heuristic(nextLocation, problem))

    return expandingPath      


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
