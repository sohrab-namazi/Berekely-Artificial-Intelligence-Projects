# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
from typing import NamedTuple, Union, Any, Sequence
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFoodList = successorGameState.getFood().asList()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        Currentfood = currentGameState.getFood().asList()

        "*** YOUR CODE HERE ***"
        nearestFood = math.inf
        for food in Currentfood:
            distToFood = manhattanDistance(food, newPos)
            if distToFood < nearestFood:
                nearestFood = distToFood
        if nearestFood == 0:
            nearestFood += 0.01
        nearestGhost = math.inf
        for ghostState in newGhostStates:
            ghostLocation = ghostState.configuration.pos
            distToGhost = manhattanDistance(newPos, ghostLocation)
            if ((distToGhost < nearestGhost) & (ghostState.scaredTimer == 0)):
                nearestGhost = distToGhost
        if nearestGhost == 0:
            return -math.inf

        point = -(1 / (nearestGhost)) + (1 / (nearestFood))
        return point


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        
        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        ghostsCount = gameState.getNumAgents() - 1
        return self.miniMax(gameState, 0, 1, ghostsCount)

    def maxValue(self, gameState, agentIndex, depth, ghostsCount):
        if (depth == self.depth + 1):
            return self.evaluationFunction(gameState)
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        maxVal = -math.inf
        maxValAction = None
        actions = gameState.getLegalActions(0)
        for action in actions:
            childState = gameState.generateSuccessor(agentIndex, action)
            maxValCandidate = self.miniMax(childState, 1, depth, ghostsCount) 
            if maxValCandidate > maxVal:
                maxVal = maxValCandidate
                maxValAction = action
        if (depth == 1):
            return maxValAction
        else:
            return maxVal    

    def minValue(self, gameState, agentIndex, depth, ghostsCount):
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        minVal = math.inf
        if agentIndex == ghostsCount:
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                childState = gameState.generateSuccessor(agentIndex, action)
                minVal = min(minVal, self.miniMax(childState, 0, depth + 1, ghostsCount))
        else:
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                childState = gameState.generateSuccessor(agentIndex, action)
                minVal = min(minVal, self.miniMax(childState, agentIndex + 1, depth, ghostsCount))
        return minVal

    def miniMax(self, gameState, agentIndex, depth, ghostsCount):
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, ghostsCount) 
        else:
            return self.minValue(gameState, agentIndex, depth, ghostsCount)  

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def isGameFinish(self, gameState):
        if (gameState.isWin()) or (gameState.isLose()):
            return True
        return False   

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ghostsCount = gameState.getNumAgents() - 1
        return self.miniMax(gameState, 0, 1, ghostsCount, -math.inf, math.inf)

    def maxValue(self, gameState, agentIndex, depth, ghostsCount, alpha, beta):
        if (depth == self.depth + 1):
            return self.evaluationFunction(gameState)
        
        maxVal = -math.inf
        maxValAction = None
        actions = gameState.getLegalActions(0)
        for action in actions:
            childState = gameState.generateSuccessor(agentIndex, action)
            maxValCandidate = self.miniMax(childState, 1, depth, ghostsCount, alpha, beta) 
            if maxValCandidate > maxVal:
                maxVal = maxValCandidate
                maxValAction = action
            if maxValCandidate > beta:
                return maxValCandidate
            alpha = max(alpha, maxValCandidate)        
        if (depth == 1):
            return maxValAction
        else:
            return maxVal    

    def minValue(self, gameState, agentIndex, depth, ghostsCount, alpha, beta):
        
        minVal = math.inf
        if agentIndex == ghostsCount:
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                childState = gameState.generateSuccessor(agentIndex, action)
                minVal = min(minVal, self.miniMax(childState, 0, depth + 1, ghostsCount, alpha, beta))
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)    
        else:
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                childState = gameState.generateSuccessor(agentIndex, action)
                minVal = min(minVal, self.miniMax(childState, agentIndex + 1, depth, ghostsCount, alpha, beta))
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)    
        return minVal

    def miniMax(self, gameState, agentIndex, depth, ghostsCount, alpha, beta):  
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState) 
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, ghostsCount, alpha, beta) 
        else:
            return self.minValue(gameState, agentIndex, depth, ghostsCount, alpha, beta)  

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        ghostsCount = gameState.getNumAgents() - 1
        return self.miniMax(gameState, 0, 1, ghostsCount)

    def maxValue(self, gameState, agentIndex, depth, ghostsCount):
        if (depth == self.depth + 1):
            return self.evaluationFunction(gameState)
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        maxVal = -math.inf
        maxValAction = None
        actions = gameState.getLegalActions(0)
        for action in actions:
            childState = gameState.generateSuccessor(agentIndex, action)
            maxValCandidate = self.miniMax(childState, 1, depth, ghostsCount) 
            if maxValCandidate > maxVal:
                maxVal = maxValCandidate
                maxValAction = action
        if (depth == 1):
            return maxValAction
        else:
            return maxVal    

    def minValue(self, gameState, agentIndex, depth, ghostsCount):
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        minVal = math.inf
        if agentIndex == ghostsCount:
            Average = 0
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                childState = gameState.generateSuccessor(agentIndex, action)
                Average += self.miniMax(childState, 0, depth + 1, ghostsCount)
            Average /= len(actions)    
        else:
            Average = 0
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                childState = gameState.generateSuccessor(agentIndex, action)
                Average += min(minVal, self.miniMax(childState, agentIndex + 1, depth, ghostsCount))
            Average /= len(actions)    
        return Average

    def miniMax(self, gameState, agentIndex, depth, ghostsCount):
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, ghostsCount) 
        else:
            return self.minValue(gameState, agentIndex, depth, ghostsCount)  

    
        


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ghosts = currentGameState.getGhostStates()
    capsulesCountMetric = len(currentGameState.getCapsules())
    foodCount = currentGameState.getNumFood()
    scoreMetric = currentGameState.getScore()
    pacmanLocation = currentGameState.getPacmanPosition()
    foodBoard = currentGameState.getFood()
    foods = foodBoard.asList()
    nearestFood = math.inf

    foods = sorted(foods, key = lambda pos: manhattanDistance(pacmanLocation, pos))
    if len(foods) > 0:
      nearestFood = foods[0]

    nearestFoodDist = 0
    if len(foods) > 0:
      nearestFoodDist = manhattanDistance(nearestFood, pacmanLocation)

    nearestFoodMetric = -nearestFoodDist

    foodCountMetric = -foodCount

    nearestGhost = math.inf
    ghostMetric = 0
    for ghost in ghosts:
      ghostPosition = ghost.getPosition()
      md = manhattanDistance(pacmanLocation, ghostPosition)
      if ghost.scaredTimer == 0:
        if md < nearestGhost:
          nearestGhost = md
      elif ghost.scaredTimer > md:
        ghostMetric += 200 - md

    if nearestGhost == math.inf:
      nearestGhost = 0
    ghostMetric += nearestGhost

    totalMetric = capsulesCountMetric + scoreMetric + ghostMetric + foodCountMetric + nearestFoodMetric

    return totalMetric
# Abbreviation
better = betterEvaluationFunction
