# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print("SuccessorGameState:", successorGameState)
        # print("newPos:", newPos)
        # print("newFood:", newFood)
        # print("newGhostStates:", newGhostStates)
        # print("newScaredTimes:", newScaredTimes)

        # if currentGameState.getPacmanPosition() == successorGameState.getPacmanPosition():
        #   return(-1) # Discourage staying in same place
        closestFood = float("inf")
        for foodPosition in newFood.asList():
            distance = util.manhattanDistance(foodPosition, newPos)
            if distance < closestFood:
                if closestFood == 0:
                    distance = 0.1
                else:
                    closestFood = distance

        ghostClose = 0
        closestGhost = float("inf")
        for ghostPosition in successorGameState.getGhostPositions():
            # if we're gonna die than avoid
            distance = util.manhattanDistance(newPos, ghostPosition)
            if distance <= 1:
                ghostClose += 1

            if distance == 0:
                closestGhost = 0.1 
            else:
                closestGhost = distance
                

        return(successorGameState.getScore() + (1 / closestFood) - (1 / closestGhost) - ghostClose)


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        def minimax(agentNum, depth, gameState):
            # HOW AM I SUPPOSED TO CALL THIS FUNCTION
            # check if depth is reached or game is won or lost. This is the basecase of our recusion.
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return(self.evaluationFunction(gameState))

            if agentNum == 0:  # For Pacman
                # https://blog.usejournal.com/how-to-work-with-infinity-in-python-337fb3987f06 that's pretty cool
                bestState = float("-inf")
                # We want to get the best state
                for action in gameState.getLegalActions(agentNum):
                    bestState = max(minimax(1, depth, gameState.generateSuccessor(
                        agentNum, action)), bestState)  # We need max for Pacman
                return(bestState)
            else:
                nextAgentNum = agentNum + 1
                if nextAgentNum >= gameState.getNumAgents():  # Check if its Pacman's turn now
                    nextAgentNum = 0
                if nextAgentNum == 0:  # Else go to the next ghost
                    depth += 1
                bestState = float("inf")
                # prospective of current agent
                for action in gameState.getLegalActions(agentNum):
                    # get min for ghosts, we want to generate successors for curent agent
                    bestState = min(minimax(
                        nextAgentNum, depth, gameState.generateSuccessor(agentNum, action)), bestState)
                return(bestState)

        # Driver function for minimax? Or else its gonna be really annoying to code this
        bestState = float("-inf")
        move = Directions.WEST  # We have to save the move this time cause we have to return it
        # We want to get the best state
        for successorState in gameState.getLegalActions(0):
            checkState = minimax(
                1, 0, gameState.generateSuccessor(0, successorState))
            if checkState > bestState:  # We need max for Pacman
                bestState = checkState
                move = successorState
        return(move)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)

      Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in AlphaBetaAgent. Again, your algorithm will be slightly more general than the pseudo-code in the textbook, so part of the challenge is to extend the alpha-beta pruning logic appropriately to multiple minimizer agents.
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaPrune(agentNum, depth, state, alpha, beta):
            if depth == self.depth or state.isLose() or state.isWin():
                # Base case. Return the value of the current state.
                return(self.evaluationFunction(state))
            return(minimax(agentNum, depth, state, alpha, beta))

        def minimax(agentNum, depth, state, alpha, beta):
            """ 
            Adapted from previous problem. Code should pretty much be the same.
            Check if depth is reached or game is won or lost. This is the basecase of our recusion.
            IDK WHY BUT IT WORKED WHEN I BROKE IT UP INTO THE FUNCTIONS ABOVE
            YAY I FIXED IT AFTER 4 HOURS OF TRYING TO FIND ONE LINE THAT WAS WRONG
            """
            if agentNum == 0:  # For Pacman we maximize
                # https://blog.usejournal.com/how-to-work-with-infinity-in-python-337fb3987f06 that's pretty cool
                bestState = float("-inf")
                # We want to get the best state
                for action in state.getLegalActions(agentNum):
                    bestState = max(bestState, alphaBetaPrune(1, depth, state.generateSuccessor(
                        agentNum, action), alpha, beta))  # We need max for Pacman
                    if bestState > beta:  # This is where we really prune
                        return(bestState)
                    alpha = max(alpha, bestState)
                return(bestState)
            else: # For ghosts we minimize 
                nextAgentNum = agentNum + 1
                if nextAgentNum == state.getNumAgents():  # Check if its Pacman's turn now
                    nextAgentNum = 0
                if nextAgentNum == 0:  # Else go to the next ghost and increase depth 
                    depth += 1
                bestState = float("inf")
                # prospective of current agent
                for action in state.getLegalActions(agentNum):
                    bestState = min(bestState, alphaBetaPrune(nextAgentNum, depth, state.generateSuccessor(
                        agentNum, action), alpha, beta))  # get min for ghosts, we want to generate successors for curent agent
                    if bestState < alpha:  # This is where we really prune
                        return(bestState)
                    beta = min(beta, bestState)
                return(bestState)

        # Driver function
        bestState = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        move = Directions.WEST  # We have to save the move this time cause we have to return it
        # We want to get the best state
        for successorState in gameState.getLegalActions(0):
            checkState = alphaBetaPrune(1, 0, gameState.generateSuccessor(
                0, successorState), alpha, beta)
            if checkState > bestState:  # We need max for Pacman
                bestState = checkState
                move = successorState
            if bestState > beta:
                return(bestState)  # I don't think this should ever happen?
            alpha = max(alpha, bestState)
        return(move)


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
        def expectimax(agentNum, depth, gameState):
            # HOW AM I SUPPOSED TO CALL THIS FUNCTION
            # check if depth is reached or game is won or lost. This is the basecase of our recusion.
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return(self.evaluationFunction(gameState))

            if agentNum == 0:  # For Pacman
                # https://blog.usejournal.com/how-to-work-with-infinity-in-python-337fb3987f06 that's pretty cool
                bestState = float("-inf")
                # We want to get the best state
                for action in gameState.getLegalActions(agentNum):
                    bestState = max(expectimax(1, depth, gameState.generateSuccessor(
                        agentNum, action)), bestState)  # We need max for Pacman
                return(bestState)
            else: # It's different for the ghost this time
                nextAgentNum = agentNum + 1
                if nextAgentNum >= gameState.getNumAgents():  # Check if its Pacman's turn now
                    nextAgentNum = 0
                if nextAgentNum == 0:  # Else go to the next ghost
                    depth += 1
                weight = 0 # We sum the weights of the moves and return the average
                # prospective of current agent
                for action in gameState.getLegalActions(agentNum):
                    # get min for ghosts, we want to generate successors for curent agent
                    weight += expectimax(
                        nextAgentNum, depth, gameState.generateSuccessor(agentNum, action))
                return(weight / len(gameState.getLegalActions(agentNum)))

        # Driver function for expectimax? Or else its gonna be really annoying to code this
        bestState = float("-inf")
        move = Directions.WEST  # We have to save the move this time cause we have to return it
        # We want to get the best state
        for successorState in gameState.getLegalActions(0):
            checkState = expectimax(
                1, 0, gameState.generateSuccessor(0, successorState))
            if checkState > bestState:  # We need max for Pacman
                bestState = checkState
                move = successorState
        return(move)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <Find food and find ghosts. Weight running away from ghost more>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foodState = currentGameState.getFood()
    foodList = foodState.asList()
    closestFood = float("inf") 

    for food in foodList:
        distance = util.manhattanDistance(food, pacmanPosition)
        if closestFood >= distance: # Get closest food 
            if closestFood == 0:
                distance = 0.1
            else:
                closestFood = distance
            

    closestGhost = float("inf")
    ghostsThatAreCloseToPacman = 0
    for ghost in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(ghost, pacmanPosition)
        if (distance < closestGhost):
            if distance == 0:
                closestGhost = 0.1 
            else:
                closestGhost = distance
        if distance <= 1:
            ghostsThatAreCloseToPacman += 1

    return(currentGameState.getScore() +  (1 / closestFood) - (1 / closestGhost) - ghostsThatAreCloseToPacman)

# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
