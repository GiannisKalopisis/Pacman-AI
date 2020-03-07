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
    return [s, s, w, s, w, w, s, w]


def notInFrontier(successor, frontier):

    for item in frontier.list:
        if item[0] == successor[0]:
            #print item[0], successor[0]
            return False
    return True


def notInFrontierUCS(successor, frontier):

    for item in list(frontier):
        if item == successor[0]:
            #print item[0], successor[0]
            return False
    return True


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    if problem.isGoalState(problem.getStartState()):
        return []

    frontier = util.Stack()
    explored = set([])
    frontier.push([problem.getStartState(), [], 0])

    while not frontier.isEmpty():

        node = frontier.pop()
        nodeState = node[0]
        nodePath = node[1]

        #if nodeState not in explored:
        #   explored.add(nodeState)
        explored.add(nodeState)
        if problem.isGoalState(nodeState):
            return nodePath


        for successor in problem.getSuccessors(node[0]):

            successorState = successor[0]
            successorPath = successor[1]

            if ((successorState not in explored) ):#and (notInFrontier(successor, frontier))):
                finalSuccessorPath = nodePath + [successorPath]

                #if problem.isGoalState(successorState):
                #    return finalSuccessorPath

                frontier.push([successorState, finalSuccessorPath, len(finalSuccessorPath)])

    #empty list is Failure
    return []
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    if problem.isGoalState(problem.getStartState()):
        return []

    frontier = util.Queue()
    explored = set([])
    # explored.add(problem.getStartState())
    frontier.push([problem.getStartState(), []])

    while not frontier.isEmpty():

        node = frontier.pop()

        # if nodeState not in explored:
        #   explored.add(nodeState)
        if problem.isGoalState(node[0]):
            return node[1]
        explored.add(node[0])


        for successor in problem.getSuccessors(node[0]):

            successorState = successor[0]
            successorPath = successor[1]


            if (successorState not in explored) and (notInFrontier(successor, frontier)):

                # if problem.isGoalState(successorState):
                #    return nodePath + [successorPath]
                # explored.add(successorState)

                #if problem.isGoalState(successorState):
                #    return nodePath + [successorPath]

                frontier.push([successorState, node[1] + [successorPath]])

    # empty list is Failure
    return []
    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    if problem.isGoalState(problem.getStartState()):
        return []

    frontier = util.PriorityQueue()

    explored = set()
    frontier.push(problem.getStartState(), 0)
    myDictForUCS ={}
    myDictForUCS[problem.getStartState()] = [[], 0]

    while not frontier.isEmpty():

        node = frontier.pop()

        if problem.isGoalState(node):
            return myDictForUCS[node][0]

        explored.add(node)

        for successor in problem.getSuccessors(node):

            if successor[0] not in explored:

                cost = myDictForUCS[node][1] + successor[2]
                path = myDictForUCS[node][0] + [successor[1]]

                if successor[0] in myDictForUCS:
                    if myDictForUCS[successor[0]][1] > cost:
                        myDictForUCS[successor[0]] = [path, cost]
                        frontier.update(successor[0], cost)
                else:
                    myDictForUCS[successor[0]] = [path, cost]
                    frontier.push(successor[0], cost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []

    frontier = util.PriorityQueue()

    explored = set()
    frontier.push(problem.getStartState(), 0)
    myDictForUCS ={}
    myDictForUCS[problem.getStartState()] = [[], heuristic(problem.getStartState(), problem)]

    while not frontier.isEmpty():

        node = frontier.pop()

        if problem.isGoalState(node):
            return myDictForUCS[node][0]

        explored.add(node)

        for successor in problem.getSuccessors(node):

            if successor[0] not in explored:

                cost = myDictForUCS[node][1] + successor[2]
                path = myDictForUCS[node][0] + [successor[1]]

                if successor[0] in myDictForUCS:
                    if myDictForUCS[successor[0]][1] > cost:
                        myDictForUCS[successor[0]] = [path, cost]
                        frontier.update(successor[0], cost + heuristic(successor[0], problem))
                else:
                    myDictForUCS[successor[0]] = [path, cost]
                    frontier.push(successor[0], cost + heuristic(successor[0], problem))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
