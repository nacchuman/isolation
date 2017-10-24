

"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

import itertools


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass



#Giving each place on the board a value in the below dectionary 
#With  max at the centre and (3,3) and goes on decreasing as we move away
place_value = {}
    
temp_row = 3
#Dividing the board into 4 blocks 
#Creating a dict of the 4 blocks the key will be a string and value will be a list 
block_groups= {}
#Creating three list to store the block elements and store the list in the above dict 
block1 = []
block2 = []
block3 = []
block4 = []
"""
    Creating place values for the board in the form 

        0   1   2   3   4   5   6

    0   1   2   3   4   3   2   1
    1   2   4   6   8   6   4   2
    2   3   6   9   12  9   6   3
    3   4   8   12  16  12  8   4
    4   3   6   9   12  9   6   3
    5   2   4   6   8   6   4   2
    6   1   2   3   4   3   2   1

    The below code creates the above plave value structure  and stores in the plave_value dict with key ( row , col )
"""



for i in range(1,8):
    temp = 3 
    for j in range(1,8):
        if j <= 4 and i <= 4 :
            place_value[(i-1,j-1)] = i*j
            #Creates a block list which includes places from (0,0) to (0,3) and (3,0) to (3,3)
            block1.append((i-1,j-1))

        elif i <= 4 and j >4 :
            place_value[(i-1,j-1)] = i * temp 
            temp = temp-1
            #Creates a block list for block 2
            block2.append((i-1,j-1))

        elif j <= 4 and i >4:
            place_value[(i-1,j-1)] = j * temp_row
            #Creates a block list for block 3
            block3.append((i-1,j-1))
        
        elif j >4 and i >4:
            place_value[(i-1,j-1)] = temp * temp_row
            temp = temp -1
            #Creates a block list for block 4 
            block4.append((i-1,j-1))



    if i > 4 :
        temp_row = temp_row -1 
#Dividing the board into 4 blocks 
block_groups["block1"] = block1
block_groups["block2"] = block2
block_groups["block3"] = block3
block_groups["block4"] = block4




def custom_score(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    score = 0.0

    for mov in  game.get_legal_moves(player):
        #Calculating the place value from the ablove created place values and returning the score 
        score += place_value[mov] 

    return score


    
def custom_score_2(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    score = 0.0
    opp_score = 0.0
    #Calculating the sum of places values for the legal moves 
    for mov in  game.get_legal_moves(player):
        score += place_value[mov] 

    #Calculating the sum of place values for legal moves for the opponent 
    for mov in game.get_legal_moves(game.get_opponent(player)):
        opp_score += place_value[mov]
        #Retruning the difference of the above created scores 
    return score - opp_score
    

def custom_score_3(game, player):


    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_score = 0.0
    opp_score = 0.0
    #Creating a list to store the empty places on the board 
    empty_places = []
    
    temp = 0 
    #Iterating over the board to find the empty places 
    for i in range(game.height):
    
            for j in range(game.width):
                idx = i + j * game.height
                if not game._board_state[idx]:
                    
                    empty_places.append((i,j))
    #Creating a dict from the same keys as block groups so as to have block values corrosponding to each 
    #group block value = sum of the place values of all the empty celss in that group 
    block_value =dict.fromkeys(block_groups.keys(),0)
    multiplier_value = dict.fromkeys(block_groups.keys(),0)
    for emp in  empty_places:
        for k , v in  block_groups.items():
            if emp in  v :
                #Adding to the block value the place value of all the empty cells in that block 
                block_value[k]  += place_value[emp]
    #Caluculating a multiplier value for each block 
    for k , v in block_value.items():
        multiplier_value[k ] = block_value[k] / len(block_groups[k])
        

    for own_mov in  game.get_legal_moves(player) :
        for  k , v in block_groups.items():
            if own_mov in v :
                own_score = place_value[own_mov] * multiplier_value[k] + own_score 
                break

        

    for opp_move in   game.get_legal_moves(game.get_opponent(player)) :
        for  k , v in block_groups.items():
            if opp_move in v : 
                opp_score = place_value[opp_move] * multiplier_value[k] + opp_score
        
    return own_score - opp_score 

    

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1,-1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move



    def min_value(self,game , depth):

        if self.time_left() < self.TIMER_THRESHOLD:

            raise SearchTimeout()

        moves = game.get_legal_moves();

        if not moves or depth == 0 :
            return self.score(game,self)

        else:
        #Assuming that the first  move is the best move and returning that in case we run out of time


            best_move_score = float('inf')


            for mov in moves:
                #implementing the move and getting the game state
                temp = game.forecast_move(mov)
                score = self.max_value(temp,depth -1 )
                if score < best_move_score:
                    best_move_score = score


        # TODO: finish this function!
        return best_move_score

    def max_value(self,game , depth):

        if self.time_left() < self.TIMER_THRESHOLD:

            raise SearchTimeout()

        moves = game.get_legal_moves()

        if not moves or depth == 0 :
            return self.score(game,self)

        else:
        #Assuming that the first  move is the best move and returning that in case we run out of time

            best_move_score = float('-inf')


            for mov in moves:
                #implementing the move and getting the game state
                temp = game.forecast_move(mov)
                score = self.min_value(temp,depth -1 )
                if score > best_move_score:
                    best_move_score = score


        # TODO: finish this function!
        return best_move_score



    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.

        """
        if self.time_left() < self.TIMER_THRESHOLD:

            raise SearchTimeout()


        moves = game.get_legal_moves();

        if not moves or depth == 0:
            return (-1,-1)

        else:
        #Assuming that the first  move is the best move and returning that in case we run out of time

            best_move = moves[0]
            best_move_score = float('-inf')


            for mov in moves:
                #implementing the move and getting the game state
                temp = game.forecast_move(mov);
                score = self.min_value(temp,depth -1 )
                if score > best_move_score:
                    best_move = mov
                    best_move_score = score


        # TODO: finish this function!
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move =(-1,-1)

        # TODO: finish this function!
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 0 
            while True:
                #print("Calling for depth ", depth)
                #print("Board looks like below we are inside get move where iterative deepnening is applied ")

                #print(game.to_string())
                best_move = self.alphabeta(game , depth)
                #print("Best Move returned ")
                depth = depth+1


        except SearchTimeout:
            #print("Serarch timeout caught ")
            #print("Depth searched ", depth)
            #print("Best move is ", best_move)
            return best_move
            # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            #print("Search time out raised in  alphabeta ")
            raise SearchTimeout()

        # TODO: finish this function!

        moves = game.get_legal_moves();

        if not moves or depth == 0:
            #print("Since depth is zero inside alpha beta returning dummy move ")
            return (-1,-1)


        #Assuming that the first  move is the best move and returning that in case we run out of time


        best_move = moves[0]

        best_move_score = float('-inf')

        #print("Depth =", depth)
        for mov in moves:
            #implementing the move and getting the game state
            #print("Calling min value from alpha beta with alpha and beta ",alpha, beta)

            best_move_score = max( best_move_score ,  self.min_value(game.forecast_move(mov),depth -1 , alpha , beta))
            #print("Best move score ", best_move_score)

            #will never be true since beta is infinity at root 
            #if best_move_score >= beta:
             #   return best_move
            # if the current value of alpha is less than the best move Score update the alpha and
            if alpha < best_move_score:
                #print("Updating alpha =", alpha)
                alpha = best_move_score
                #print("Updating best move =", best_move)
                best_move = mov
        # TODO: finish this function!

            alpha = max(alpha , best_move_score )
            #print("Alpha ",alpha)
        return best_move


    def max_value(self, game , depth , alpha , beta ):


        if self.time_left() < self.TIMER_THRESHOLD:
            #print("Search time out raised in max value ")
            raise SearchTimeout()
        moves = game.get_legal_moves()


        if not moves or depth == 0 :
            #print("Returning score at depth",depth,"Inside max")
            #print("game looks like this ", game.to_string())
            #print("Score returned is ", self.score(game,self))

            return self.score(game,self)




        #Assuming that the first  move is the best move and returning that in case we run out of time

        best_move_score = float('-inf')

        #print("Total moves possible", len(moves))
        for mov in moves:
            #implementing the move and getting the game state
            #print("Evaluating for move =",mov)
            best_move_score =max( best_move_score ,  self.min_value(game.forecast_move(mov),depth -1 , alpha , beta))

            if best_move_score >= beta:
                #print("Best move score is greater than beta ")
                #print("beta=", beta)
                #print("Returning best move score ", best_move_score)
                return best_move_score

            alpha = max(alpha , best_move_score )
            #print("Alpha is =",alpha)
        # TODO: finish this function!
        return best_move_score

    def min_value(self, game , depth , alpha , beta ):


        if self.time_left() < self.TIMER_THRESHOLD:
            #print("Search time out raised in min value ")

            raise SearchTimeout()

        moves = game.get_legal_moves()

        if not moves or depth == 0 :
            #print("Returning score at depth",depth,"Inside min")
            #print(game.to_string())
            #print("Score returned is ", self.score(game,self))
            return self.score(game,self)

        #assuming worst possible case for best move score that is - inf 
        best_move_score = float('inf')

        #print("Total moves ", len(moves))
        for mov in moves:
        #implementing the move and getting the game state

            #print("Move under evalutaion", mov)
            best_move_score =min( best_move_score ,  self.max_value(game.forecast_move(mov) ,depth -1 , alpha , beta) )
            #print("Best Move score returned  =", best_move_score)
            if best_move_score <= alpha :
                #print("prunning since best move score ", best_move_score , "Is less than alpha ",alpha)
                return best_move_score

            beta = min(beta , best_move_score )
            #print("Beta is =", beta)

        # TODO: finish this function!
        return best_move_score

