import pygame
import argparse
import time
from random import shuffle, seed, sample
from copy import deepcopy
from heapq import heappush, heappop

# board parameters
WIDTH = 600
HEIGHT = 600 
INSTRUCTIONS_HEIGHT = 100
FPS = 60
GAME_NAME = "N-Puzzle"

# colors
WHITE=(255,255,255)
NAVY_BLUE = (0, 0, 128)
BLACK = (0,0,0)
GREY = (128, 128, 128)

# fonts
pygame.init() # to avoid font error
FONT1 = pygame.font.SysFont("comicsans", 30)


class Tile:
   

    def __init__(self, val, x, y, w, l, n):
        self.val = val
        self.x = x
        self.y = y
        self.w = w
        self.l = l
        self.board_size = n

    def draw_tile(self, window):
        #it calculates the appropriate font size and position based on the size of the game board
        #Different font sizes and positions are used for different board sizes to ensure proper rendering
        val = self.val
        if val == 0:
            pygame.draw.rect(window, GREY, (self.x, self.y, self.w, self.l))
        else:
            
            # manual calibrations for board sizes
            if self.board_size <= 4:
                font = pygame.font.SysFont('arial', 100)
                if int(val) >= 10:
                    draw_x = self.x + self.w // 7 
                    draw_y = self.y + self.l // 4
                else:
                    draw_x = self.x + self.w // 3 
                    draw_y = self.y + self.l // 4
            
            elif self.board_size == 5:
                font = pygame.font.SysFont('arial', 80)
                if int(val) >= 10:
                    draw_x = self.x + self.w // 9 
                    draw_y = self.y + self.l // 6
                else:
                    draw_x = self.x + self.w // 3 
                    draw_y = self.y + self.l // 6
 
            elif self.board_size == 6:
                font = pygame.font.SysFont('arial', 60)
                if int(val) >= 10:
                    draw_x = self.x + self.w // 9 
                    draw_y = self.y + self.l // 6
                else:
                    draw_x = self.x + self.w // 3 
                    draw_y = self.y + self.l // 6

            else:
                #handling error if the size of the board exceeds 6
                raise Exception("Not calibrated for board size: {}"\
                    .format(self.board_size))

            text = font.render(str(val), True, WHITE)
            window.blit(text, (draw_x, draw_y))
                    
    def __repr__(self):
        return str(self.val)


class Board:

    OUTER_BORDER_SIZE = 10
    INNER_BORDER_SIZE = 5
        
    MOVE_DIRS = { 
                  "RIGHT" : (0, 1),
                  "LEFT"  : (0, -1),
                  "DOWN"  : (1, 0),
                  "UP"    : (-1, 0)
                }

    def __init__(self, n, board=None, random_shifts=1000, board_prints=True):
       
        self.board_prints = board_prints
 
        self.board = board if board != None else [i for i in range(n*n)]
        self.rows = n
        self.cols = n
        
        # inner borders
        self.inner_width  = WIDTH - 2 * self.OUTER_BORDER_SIZE - \
                                (self.cols - 1) * self.INNER_BORDER_SIZE
        self.inner_height = HEIGHT - 2 * self.OUTER_BORDER_SIZE - \
                                (self.rows - 1) * self.INNER_BORDER_SIZE
        self.tile_width = self.inner_width / self.cols
        self.tile_height = self.inner_height / self.rows
      
        self.tiles = []
        for row in range(self.rows):
            self.tiles.append([])
            for col in range(self.cols):
                
               #each tile know it's cordinates
                self.tiles[row].append(Tile(self.board[row*self.cols + col], 
                    self.OUTER_BORDER_SIZE + \
                        col * (self.tile_width + self.INNER_BORDER_SIZE),
                    self.OUTER_BORDER_SIZE + \
                        row * (self.tile_height + self.INNER_BORDER_SIZE), 
                        self.tile_width, self.tile_height, self.rows))

        # to ensure our board has a solution, we start with the solved state
        # and randomize it according to legal moves. This is important, since
        # not all permuations of N-puzzles have solutions
        # progress our board using legal moves, our board will have a solution
        all_pos = list(self.MOVE_DIRS.keys())
        for shift in range(random_shifts):
            move = sample(all_pos, 1)[0]
            self.forecast_move(move)
        print(self.tiles)

    def draw(self, window):
        window.fill(NAVY_BLUE)
        
        # horizontal outer borders
        pygame.draw.rect(window, BLACK, (0, 0, WIDTH, self.OUTER_BORDER_SIZE)) 
        pygame.draw.rect(window, BLACK, (0, HEIGHT - self.OUTER_BORDER_SIZE, \
                         WIDTH, self.OUTER_BORDER_SIZE)) 
        
        # vertical outer borders
        pygame.draw.rect(window, BLACK, (0, 0, self.OUTER_BORDER_SIZE, HEIGHT))
        pygame.draw.rect(window, BLACK, (WIDTH - self.OUTER_BORDER_SIZE, 0,  \
                         self.OUTER_BORDER_SIZE, HEIGHT))
 
        for col in range(1, self.cols):
            pygame.draw.rect(window, BLACK, (self.OUTER_BORDER_SIZE + col * \
                self.tile_width + (col-1) * self.INNER_BORDER_SIZE, 0, \
                self.INNER_BORDER_SIZE, HEIGHT)) 
            
        for row in range(1, self.rows):
            pygame.draw.rect(window, BLACK, (0, self.OUTER_BORDER_SIZE + row  *\
                self.tile_height + (row-1) * self.INNER_BORDER_SIZE, WIDTH, \
                self.INNER_BORDER_SIZE)) 

        for row in range(self.rows):
            for col in range(self.cols):
                self.tiles[row][col].draw_tile(window) 


    def forecast_move(self, move):
        # It checks whether the specified move is valid and updates the position of the empty tile accordingly        
        if self.board_prints:
            print(move)
 
        # find the empty tile location
        zero_pos = None
        for row in range(self.rows):
            for col in range(self.cols):
                if self.tiles[row][col].val == 0:
                    zero_pos = (row, col)
                    break   
            if zero_pos != None:
                break

        assert zero_pos != None

        if not move in self.MOVE_DIRS:
            raise Exception ("Illegal move name passed to forecast_move: {}".format(move))
       
        # compute new locations
        x, y = self.MOVE_DIRS[move]
        row, col = zero_pos
        row2, col2 = row + x, col + y
        
        if row2 < 0 or col2 < 0 or row2 >= self.rows or col2 >= self.cols:
            if self.board_prints:
                print("Impossible move passed") 
            return False
        else:
            self.tiles[row][col].val = self.tiles[row2][col2].val
            self.tiles[row2][col2].val = 0
            return True

import abc
from copy import deepcopy
from heapq import heappush, heappop

class AStarBaseHeuristicComputer():
    
    def __init__(self):
        pass

    @abc.abstractmethod
    def compute_heuristic(self, tiles):
        pass


class AStarManhattanHeuristic():

    def __init__(self):
        pass

    def compute_heuristic(self, tiles):
        # returns sum of manhattan distances to the correct locations
        d = {}
        n = len(tiles)
        for i in range(n):
            for j in range(n):
                val = tiles[i][j].val
                d[val] = (i, j)

        summ = 0
        for val in range(n * n):
            row_diff = abs(val // n - d[val][0]) 
            col_diff = abs(val %  n - d[val][1]) 
            summ += row_diff + col_diff
        return summ


class AStarManhattanHeuristicOuterEmphasis():
    '''additional weight given to tiles located at the outer edges of the puzzle grid
This additional weight helps to prioritize moves that bring tiles closer to their correct positions on the outer edges'''

    def __init__(self):
        pass

    def compute_heuristic(self, tiles):
        # returns sum of manhattan distances to the correct locations
        d = {}
        n = len(tiles)
        for i in range(n):
            for j in range(n):
                val = tiles[i][j].val
                d[val] = (i, j)

        summ = 0
        for val in range(n * n):
            row_diff = abs(val // n - d[val][0]) * (val // n)
            col_diff = abs(val %  n - d[val][1]) * (val % n)
            summ += row_diff + col_diff
        return summ


class UnidirectionalSolver:
    
    def __init__(self, board, heuristic=AStarManhattanHeuristic):
        self.board = board
        self.heuristic_computer = heuristic()
        print("Using solver: {}".format(self.name()))


    def get_tile_tup(self, board):
        #generates a tuble representing the current state of the game board by iterating on each tile and appending it's value to the tuple
        
        tup = []
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                tup.append(board.tiles[row][col].val)
        tup = tuple(tup)
        return tup    

    
    def get_solution(self):
        
        visited = set([])#initializing a set  of the visited nodes to keep track of it
        new_h = self.heuristic_computer.compute_heuristic(self.board.tiles)
        f = [((new_h, new_h, [], deepcopy(self.board)))]#ensuring that the search begins from the starting state of the game board with the appropriate heuristic value and an empty path.







        
        start_time = time.time()  # Start the timer
        
        while(len(f) > 0):

            h, cost, path, board = heappop(f)

            if h == 0:
                return len(visited), path, time.time() - start_time  # Stop the timer and return the elapsed time

            tup = self.get_tile_tup(board)
            if tup in visited:
                continue
            visited.add(tup)
 
            for move in ["RIGHT", "LEFT", "UP", "DOWN"]:#expansion to explore everypossible move of the current state of the board
            #If the move is valid and the resulting board state has not been visited before, we compute the heuristic value for the new state and enqueue it into f.
                new_board = deepcopy(board)
                if new_board.forecast_move(move) and \
                                not self.get_tile_tup(new_board) in visited:
                    new_h = self.heuristic_computer.compute_heuristic(\
                                                new_board.tiles)
                    heappush(f, (new_h, new_h, path + [move], new_board))

    
    def name(self):
        return type(self).__name__ + '_' \
            + type(self.heuristic_computer).__name__
            
        '''It iteratively explores states with the lowest heuristic values
       generating new states from valid moves and adding them to the priority queue for further exploration'''     
            
    
     







import time
import pygame 
import argparse




WINDOW = pygame.display.set_mode((WIDTH, HEIGHT + INSTRUCTIONS_HEIGHT))

pygame.display.set_caption(GAME_NAME)


def parse_cli():
    parser = argparse.ArgumentParser(
            description="Argument Parser for N-Puzzle Game")
    parser.add_argument("-n", "-n_size", type=int, default=3)
    return parser.parse_args()


def instructions(window, automatic_solve_invoked=False):
            
    pygame.draw.rect(window, NAVY_BLUE, (0, HEIGHT, WIDTH, INSTRUCTIONS_HEIGHT))
    
    if automatic_solve_invoked:
        text1 = FONT1.render(\
            "AI is taking care of it now..",\
            1, WHITE) 
        window.blit(text1, (10, HEIGHT + 10))         
    else: 
        text1 = FONT1.render("use the arrow to play",\
            1, WHITE) 
        text2 = FONT1.render("press a for ai to solve it",\
            1, WHITE) 
        window.blit(text1, (10, HEIGHT + 10))         
        window.blit(text2, (10, HEIGHT + 50))         


def solved_instructions(window, step, total, total_visited, time_elapsed):
    pygame.draw.rect(window, NAVY_BLUE, (0, HEIGHT, WIDTH, INSTRUCTIONS_HEIGHT))
    
    text1 = FONT1.render("AI is taking care of it now..",\
        1, WHITE) 

    text2 = FONT1.render("found solution after visiting {} nodes"\
        .format(total_visited), 1, WHITE) 

    text3 = FONT1.render("Move {} out of {}"\
        .format(step, total),\
        1, WHITE) 

    text4 = FONT1.render("Time Elapsed: {:.2f} seconds".format(time_elapsed), 1, WHITE)

    window.blit(text1, (10, HEIGHT + 10))         
    window.blit(text2, (10, HEIGHT + 40))         
    window.blit(text3, (10, HEIGHT + 70))
    window.blit(text4, (10, HEIGHT + 100))         


def main():

    run = True
    clock = pygame.time.Clock()

    args = parse_cli()
    board = Board(args.n)
    
    while run:
        
        clock.tick(FPS)
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pass
       
            elif event.type == pygame.KEYDOWN: 
            
                if event.key == pygame.K_DOWN:
                    board.forecast_move("DOWN")
            
                elif event.key == pygame.K_UP:
                    board.forecast_move("UP")
        
                elif event.key == pygame.K_LEFT:
                    board.forecast_move("LEFT")
            
                elif event.key == pygame.K_RIGHT:
                    board.forecast_move("RIGHT")
           
                elif event.key == pygame.K_a:
                    
                    # first, update display to reflect current course of action
                    instructions(WINDOW, automatic_solve_invoked=True)
                    pygame.display.update()
                   
                    # get solution 
                    solver = UnidirectionalSolver(board)
                    num_visited, moves, time_elapsed = solver.get_solution()
                    print("Solution found after visiting {} nodes".format(\
                        num_visited))                   
 
                    # play back the solution, printing each step
                    for ind, move in enumerate(moves):
                        print("Move {} of {}".format(ind, len(moves)))
                        board.forecast_move(move)
                        board.draw(WINDOW)
                        solved_instructions(WINDOW, ind, len(moves), num_visited, time_elapsed)
                        pygame.display.update()
                        time.sleep(0.25)
                    time.sleep(5)
                else:
                    print("I don't care about that key ;)")
            else:
                pass
        
        board.draw(WINDOW)
        instructions(WINDOW)
        pygame.display.update()

    pygame.quit()
        

if __name__ == "__main__":
    main()
