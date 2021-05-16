# get user agent from https://www.whoishostingthis.com/tools/user-agent/
import time
import numpy as np
import minimaxHeuristics as mh

def main():
    n = int(input('Enter baord size: '))
    m = int(input('Enter target size: '))
    
    # game starts here
    win = None
    board = createBoard(n)
    all_moves = generateAllMoves(n)
    
    if n <= 15:
        move_limits = 120 # if below this limit, it will run all possible moves
        priority_moves = 150
    elif n <= 18:
        move_limits = 120
        priority_moves = 200
    else: 
        move_limits = 100
        priority_moves = 250
    
    agent1 = mh.TTT(3, 0.9, "X", "O", move_limits, priority_moves, m, n*n)
    agent2 = mh.TTT(3, 0.9, "O", "X", move_limits, priority_moves, m, n*n)

    
    mymove = all_moves[0]
    tt = 0

    # continues game until there's a winner
    while win == None:
        start = time.time()
        mymove, board, all_moves = agent1.findBestMove(board, all_moves, mymove)
        end = time.time()
        tt = end - start
        print(board)
        print("X Remaining moves: ", len(all_moves), " Time taken: ", tt, "\nCurrent move: ", mymove, "\n")
        win = mh.winner(board, m, len(all_moves) == 0, mymove, "X")
        if win != None:
            break

        start = time.time()
        mymove, board, all_moves = agent2.findBestMove(board, all_moves, mymove)
        end = time.time()
        tt = end-start
        win = mh.winner(board, m, len(all_moves) == 0, mymove, "O")
        print(board)
        print("O Remaining moves: ", len(all_moves), " Time taken: ", tt, "\nCurrent move: ", mymove, "\n")
        print()
    
    print("Winner ", win)
    return

def createBoard(n):
    board = [['-' for i in range(n)] for j in range(n)] # create a 2D array
    return np.array(board)

def generateAllMoves(n):
    temp = [[False for j in range(n) ] for i in range(n)]
    all_moves = []
    i, j, di, dj = 0, 0, 0, 1
    for x in range(n*n):
        all_moves.insert(0, (i, j))
        temp[i][j] = True
        if temp[(i+di)%n][(j+dj)%n] == True:
            di, dj = dj, -di
        i += di
        j += dj
    return all_moves

if __name__ == "__main__":
    main()
