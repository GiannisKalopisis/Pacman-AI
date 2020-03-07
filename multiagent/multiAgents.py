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

def myManhattanDistance(position1, position2):
    xy1 = position1
    xy2 = position2
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print("Score is: ", scores)
        bestScore = max(scores)
        #print("BestScore is: ", bestScore)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #print("BestIndices is: ", bestIndices)
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print("ChosenIndex is: ", chosenIndex)

        "Add more of your code here if you want to"
        #print("LegalMoves is: ", legalMoves)
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        #=== food ===
        numOfFoods = len(newFood.asList())
        foodDistances = set()
        minFoodDistance = 0
        for food in currentGameState.getFood().asList():	#newFood.asList()
            foodDistances.add(myManhattanDistance(newPos, food))
        minFoodDistance = min(foodDistances)

        #=== capsule ===
        numOfCapsules = len(successorGameState.getCapsules())

        #=== ghosts ===

        activeGhostsDistance = set()
        minActiveGhost = 0
        deactivatedGhostsDistance = set()
        minDeactivatedGhostsDistance = 0
        for ghost in newGhostStates:
            if ghost.scaredTimer <= 0:
                activeGhostsDistance.add(myManhattanDistance(newPos, ghost.getPosition()))
            else:
                deactivatedGhostsDistance.add(myManhattanDistance(newPos, ghost.getPosition()))

        if len(activeGhostsDistance) > 0:
            minActiveGhost = min(activeGhostsDistance)
        if len(deactivatedGhostsDistance) > 0:
            minDeactivatedGhostsDistance = min(deactivatedGhostsDistance)


        if minActiveGhost < 2:
            return -10000


        return minFoodDistance * -1 + \
               minActiveGhost * 1/2 + \
               minDeactivatedGhostsDistance * -5 + \
               numOfCapsules * -1.5 + \
               numOfFoods * -2




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
        """
        "*** YOUR CODE HERE ***"
        #we are at pacman-max state (root)

        n = -1000000
        #print n
        highestAction = None

        for action in gameState.getLegalActions(0): #0 is Pacman

            costDirection = self.MinValue(gameState.generateSuccessor(0, action), 0, 1)
            #print "==>", costDirection

            if costDirection > n:
                n = costDirection
                highestAction = action

        return highestAction



    def MaxValue(self, gameState, depth):

        #terminal state:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        n = -1000000

        costDirection = None

        for action in gameState.getLegalActions(0): #0 is Pacman
            succ = gameState.generateSuccessor(0, action)
            costDirection = self.MinValue(succ, depth, 1)
            #print "-->", costDirection

            if costDirection > n:
                n = costDirection

        return n



    def MinValue(self, gameState, depth, index):

        # terminal state:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        n = 1000000

        costDirection = None


        for action in gameState.getLegalActions(index):
            succ = gameState.generateSuccessor(index, action)
            if ((index) % (gameState.getNumAgents() - 1)) != 0:
                costDirection = self.MinValue(succ, depth, index + 1)
            else:
                costDirection = self.MaxValue(succ, depth + 1)


            if costDirection < n:
                #print "n: ", n, ", costDirection: ", costDirection
                n = costDirection

        return n


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # we are at pacman-max state (root)

        n = -1000000
        a = -1000000
        b = 1000000
        # print n
        highestAction = None

        for action in gameState.getLegalActions(0):  # 0 is Pacman

            costDirection = self.MinValue(gameState.generateSuccessor(0, action), 0, 1, a, b)
            # print "==>", costDirection

            if costDirection > n:
                n = costDirection
                highestAction = action

            a = max(a, n)
            if n > b:
                #return highestAction
                continue

        return highestAction


    def MaxValue(self, gameState, depth, a, b):

        # terminal state:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        n = -1000000

        costDirection = None

        for action in gameState.getLegalActions(0):  # 0 is Pacman

            succ = gameState.generateSuccessor(0, action)

            costDirection = self.MinValue(succ, depth, 1, a, b)
            # print "-->", costDirection

            if costDirection > n:
                n = costDirection
            #print "n: ", n

            a = max(a, n)
            if n > b:
                return n

            #print "a: ", a
            #print "b: ", b

        return n

    def MinValue(self, gameState, depth, index, a, b):

        # terminal state:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        n = 1000000

        costDirection = None

        for action in gameState.getLegalActions(index):

            succ = gameState.generateSuccessor(index, action)

            if ((index) % (gameState.getNumAgents() - 1)) != 0:
                costDirection = self.MinValue(succ, depth, index + 1, a, b)
            else:
                costDirection = self.MaxValue(succ, depth + 1, a, b)

            if costDirection < n:
                # print "n: ", n, ", costDirection: ", costDirection
                n = costDirection
            #print "-->n: ", n

            if n < a:
                return n
            b = min(b, n)

            #print "-->a: ", a
            #print "-->b: ", b


        return n


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
        # we are at pacman-max state (root)

        n = -1000000
        # print n
        highestAction = None

        for action in gameState.getLegalActions(0):  # 0 is hero (Pacman)

            costDirection = self.MinValue(gameState.generateSuccessor(0, action), 0, 1)
            # print "==>", costDirection

            if costDirection > n:
                n = costDirection
                highestAction = action

        return highestAction

    def MaxValue(self, gameState, depth):

        # terminal state:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        n = -1000000

        costDirection = None

        for action in gameState.getLegalActions(0):  # 0 is Pacman
            succ = gameState.generateSuccessor(0, action)
            costDirection = self.MinValue(succ, depth, 1)
            # print "-->", costDirection

            if costDirection > n:
                n = costDirection

        return n

    def MinValue(self, gameState, depth, index):

        # terminal state:
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        n = 1000000
        sum = 0.0
        counter = 0.0
        costDirection = None

        for action in gameState.getLegalActions(index):
            succ = gameState.generateSuccessor(index, action)
            if ((index) % (gameState.getNumAgents() - 1)) != 0:
                costDirection = self.MinValue(succ, depth, index + 1)
            else:
                costDirection = self.MaxValue(succ, depth + 1)

            #if costDirection < n:
                # print "n: ", n, ", costDirection: ", costDirection
            #    n = costDirection
            sum += float(costDirection)
            counter += 1.0

        return sum/counter


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    if currentGameState.isWin():
        return 1000000

    state = currentGameState.getPacmanState()
    newPos = state.getPosition()
    #print newPos
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # === food ===
    numOfFoods = len(newFood.asList())
    foodDistances = set()
    minFoodDistance = 0
    for food in currentGameState.getFood().asList():
        foodDistances.add(myManhattanDistance(newPos, food))
    minFoodDistance = min(foodDistances)

    # === capsule ===
    numOfCapsules = len(currentGameState.getCapsules())

    # === ghosts ===

    activeGhostsDistance = set()
    minActiveGhost = 0
    deactivatedGhostsDistance = set()
    minDeactivatedGhostsDistance = 0
    for ghost in newGhostStates:
        if ghost.scaredTimer <= 0:
            activeGhostsDistance.add(myManhattanDistance(newPos, ghost.getPosition()))
        else:
            deactivatedGhostsDistance.add(myManhattanDistance(newPos, ghost.getPosition()))

    if len(activeGhostsDistance) > 0:
        minActiveGhost = min(activeGhostsDistance)
    if len(deactivatedGhostsDistance) > 0:
        minDeactivatedGhostsDistance = min(deactivatedGhostsDistance)

    if minActiveGhost < 1:
        return -10000


    return minFoodDistance * -1 + \
           minActiveGhost * 1 / 2 + \
           minDeactivatedGhostsDistance * -5 + \
           numOfCapsules * -20 + \
           numOfFoods * -25




# Abbreviation
better = betterEvaluationFunction

