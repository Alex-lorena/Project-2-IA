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
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
e
        Print out these variables to se what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0

        for ghostpos in successorGameState.getGhostPositions():
            dist = manhattanDistance(newPos, ghostpos)
            if dist < 2:
                score -= 1000  
            score -= 10.0

        foodList = newFood.asList()
        if len(foodList) == 0:
            return score + 1000  

        closestFood = min(manhattanDistance(newPos, f) for f in foodList)
        score += 10.0 / closestFood  

        if action == Directions.STOP:
            score -= 50.0

        return successorGameState.getScore() + score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            
            if agentIndex == 0:
                bestScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = minimax(1, depth, successor)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action

                return bestScore, bestAction
            
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    nextDepth += 1

                bestScore = float('inf')
                
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = minimax(nextAgent, nextDepth, successor)
                    bestScore = min(bestScore, score)

                return bestScore, None

        _, bestMove = minimax(0, 0, gameState)
        if bestMove is None:
            legal = gameState.getLegalActions(0)
            return legal[0]
    
        return bestMove

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:
                bestScore = float('-inf')
                bestAction = None

                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = alphaBeta(1, depth, successor, alpha, beta)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action

                    if bestScore > beta:
                        break  

                    alpha = max(alpha, bestScore)

                return bestScore, bestAction

            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == gameState.getNumAgents():  
                    nextAgent = 0
                    nextDepth += 1

                bestScore = float('inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                    if score < bestScore:
                        bestScore = score
                        bestAction = action

                    if bestScore < alpha:
                        break  

                    beta = min(beta, bestScore)

                return bestScore, bestAction

        _, action = alphaBeta(0, 0, gameState, float('-inf'), float('inf'))
        if action is None:
            legal = gameState.getLegalActions(0)
            return legal[0]
        return action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            
            legalActions = gameState.getLegalActions(agentIndex)
            if len(legalActions) == 0:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:
                bestscore = float('-inf')
                bestAction = None
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = expectimax(1, depth, successor)
                    if score > bestscore:
                        bestscore = score
                        bestAction = action

                return bestscore, bestAction

            else:
                nextIndex = agentIndex + 1
                nextDepth = depth
                if nextIndex == gameState.getNumAgents():
                    nextIndex = 0
                    nextDepth = depth + 1

                probability = 1.0 / len(legalActions)
                expectedscore = 0
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = expectimax(nextIndex, nextDepth, successor)
                    expectedscore += probability * score

                return expectedscore, None

        _, bestAction = expectimax(0, 0, gameState)
        if bestAction is None:
            legal = gameState.getLegalActions(0)
            return legal[0]
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    
    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    if len(foodList) == 0:
        return float('inf')
    
    if foodList:
        foodDistances = [manhattanDistance(pos, f) for f in foodList]
        closestFood = min(foodDistances)
        score += 40.0 / (closestFood + 1.0)
        score -= 10.0 * len(foodList)
            

    for ghostState, ghostPos in zip(ghostStates, ghostPositions):
        d = manhattanDistance(pos, ghostPos)

        if ghostState.scaredTimer > 0:  
            score += 200.0 / (d + 1)
        else:                            
            if d <= 1:
                return float('-inf')     
            elif d <= 2:
                score -= 200
            else:
                score -= 10.0 / (d + 1.0)

    score -= 20 * len(capsules)
    for cap in capsules:
        score += 10.0 / (manhattanDistance(pos, cap) + 1)
    
    if pos == Directions.STOP:
            score -= 500

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
