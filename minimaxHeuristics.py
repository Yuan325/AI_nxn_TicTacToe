import numpy as np
# board (2D array), n, m, sign (X or O), all_moves (list of moves)
# remember previous utility calculations (to reduce computation)
class TTT:
    def __init__(self, depth, gamma, sign, opponent, move_limits, prior_limits, m, reward):
        self.MAX_DEPTH = depth
        self.GAMMA = gamma
        self.MY_SIGN = sign
        self.OPPO_SIGN = opponent
        self.MOVE_LIMITS = move_limits
        self.PRIORITY_LIMITS = prior_limits
        self.M = m
        self.REWARD = reward

    def findBestMove(self, board, all_moves, prev ):
        l, r, t, b = getCornerVal(len(board), int(self.M/2), prev[0], prev[1]) 
        best_u = float('-inf') 
        best_move = (-1, -1) 
        count = 0
        for i, move in enumerate(all_moves):
            if len(all_moves) <= self.MOVE_LIMITS or count <= self.PRIORITY_LIMITS or (move[0] >= t and move[0] <= b and move[1] >= l and move[1] <= r):
                all_moves.remove(move)
                board[move[0]][move[1]] = self.MY_SIGN
                u = self.minimax(board, all_moves, 1, move, self.MY_SIGN, float('-inf'), float('inf')) 
                #print(move, " ", u)
                if u > best_u:
                    best_u = u
                    best_move = move
                all_moves.insert(i, move)
                if u == self.REWARD:
                    best_move = move
                    break
                board[move[0]][move[1]] = "-"
                count += 1
        board[best_move[0]][best_move[1]] = self.MY_SIGN
        all_moves.remove(best_move)
        return best_move, board, all_moves

    # even number depth = my turn
    # odd number depth = opponent's turn
    def minimax(self, board, all_moves, depth, move, prev, alpha, beta):
        win = winner(board, self.M, len(all_moves) == 0, move, prev)
        if win != None: 
            if win == self.MY_SIGN:
                return self.REWARD
            if win == self.OPPO_SIGN:
                return -self.REWARD
            else:
                return 0 

        if depth == self.MAX_DEPTH:
            u = self.estimate_u(board)
        elif prev == self.OPPO_SIGN:
            u = self.maxValue(board, all_moves, depth, alpha, beta)
        else:
            u = self.minValue(board, all_moves, depth, alpha, beta)
        return u * self.GAMMA

    def maxValue(self, board, all_moves, depth, alpha, beta):
        v = float('-inf')
        for i, move in enumerate(all_moves):
            all_moves.remove(move)
            board[move[0]][move[1]] = self.MY_SIGN
            v = max(v, self.minimax(board, all_moves, depth+1, move, self.MY_SIGN, alpha, beta))
            all_moves.insert(i, move)
            board[move[0]][move[1]] = "-"
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, board, all_moves, depth, alpha, beta):
        v = float('inf')
        for i, move in enumerate(all_moves):
            all_moves.remove(move)
            board[move[0]][move[1]] = self.OPPO_SIGN
            v = min(v, self.minimax(board, all_moves, depth+1, move, self.OPPO_SIGN, alpha, beta))
            all_moves.insert(i, move)
            board[move[0]][move[1]] = "-"
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v


    # estimate the utilities for current board
    # higher point for most likely to win
    # lower point for most likely to lose
    def estimate_u(self, board): 
        u = 0
        empty = 0
        temp = 0
        cur_symb = []
        cur = ""
        per_symb = 1/self.M

        # check rows
        for row in board:
            if "X" in row or "O" in row:
                cur = ""
                cur_symb = []
                empty= 0
                temp = 0
                for i, symb in enumerate(row):
                    if symb == "-":
                        empty += 1
                        temp -= 1
                    else:
                        if symb != cur:
                            cur_symb = []
                        cur = symb

                        if len(cur_symb) == 0:
                            temp = self.M - empty - 1
                        else:
                            temp -= 1
                        empty = 0
                        
                        cur_symb.append( i+self.M-1 )
                  
                    if (temp <= 0) and cur != "":
                        # Reward need to be changed to 2**n
                        #u = u + (2**len(cur_symb)) if cur == self.MY_SIGN else u - (2**len(cur_symb))
                        u = u + (per_symb * len(cur_symb)) if cur == self.MY_SIGN else u - (per_symb * len(cur_symb))
                        if len(cur_symb) + 1 == self.M:
                            u = u + len(board) if cur == self.MY_SIGN else u - (len(board)*2)
                        #u = u + (2**len(cur_symb)) if cur == self.MY_SIGN else u - (2**(len(cur_symb) + 2))
                    if len(cur_symb) != 0 and cur_symb[0] == i:
                        cur_symb.pop(0)
        
        # check column
        for col in board.T:
            if "X" in col or "O" in col:
                cur = ""
                cur_symb = []
                empty= 0
                temp = 0
                for i, symb in enumerate(col):
                    if symb == "-":
                        empty += 1
                        temp -= 1
                    else:
                        if symb != cur:
                            cur_symb = []
                        cur = symb

                        if len(cur_symb) == 0:
                            temp = self.M - empty - 1
                        else:
                            temp -= 1
                        empty = 0
                        
                        cur_symb.append( i+self.M-1 )
                  
                    if (temp <= 0) and cur != "":
                        #u = u + (2**len(cur_symb)) if cur == self.MY_SIGN else u - (2**len(cur_symb))
                        u = u + (per_symb * len(cur_symb)) if cur == self.MY_SIGN else u - (per_symb * len(cur_symb))
                        if len(cur_symb) + 1 == self.M:
                            u = u + len(board) if cur == self.MY_SIGN else u - (len(board)*2)
                        #u = u + (2**len(cur_symb)) if cur == self.MY_SIGN else u - (2**(len(cur_symb) + 2))
                    if len(cur_symb) != 0 and cur_symb[0] == i:
                        cur_symb.pop(0)
        # check diagonal1
        n = len(board)
        end = n - self.M + 1
        temp_s = []
        temp_s.append((0, 0))
        for i in range(1, end):
            temp_s.append((0, i))
            temp_s.append((i, 0))

        for i, loc in enumerate(temp_s):
            cur_i = loc[0]
            cur_j = loc[1]
            cur = ""
            cur_symb = []
            empty= 0
            temp = 0
            while cur_i < n and cur_j < n:
                symb = board[cur_i][cur_j]
                if symb == "-":
                    empty += 1
                    temp -= 1
                else:
                    if symb != cur:
                        cur_symb = []
                    cur = symb

                    if len(cur_symb) == 0:
                        temp = self.M - empty - 1
                    else:
                        temp -= 1
                    empty = 0
                    
                    cur_symb.append( i + self.M - 1 )
                if (temp <= 0) and cur != "":
                    #u = u + (2**len(cur_symb)) if cur == self.MY_SIGN else u - (2**len(cur_symb))
                    u = u + (per_symb * len(cur_symb)) if cur == self.MY_SIGN else u - (per_symb * len(cur_symb))
                    if len(cur_symb) + 1 == self.M:
                        u = u + len(board) if cur == self.MY_SIGN else u - (len(board)*2)
                    #u = u + (2**len(cur_symb)) if cur == self.MY_SIGN else u - (2**(len(cur_symb) + 2))
                if len(cur_symb) != 0 and cur_symb[0] == i:
                    cur_symb.pop(0)
                cur_i += 1
                cur_j += 1

        # check diagonal 2
        temp_s = []
        for i in range(end, n):
            temp_s.append((0, i))
        for i in range(1, end):
            temp_s.append((i, n-1))

        for i, loc in enumerate(temp_s):
            cur_i = loc[0]
            cur_j = loc[1]
            cur = ""
            cur_symb = []
            empty= 0
            temp = 0
            while cur_i < n and cur_j >= 0:
                symb = board[cur_i][cur_j]
                if symb == "-":
                    empty += 1
                    temp -= 1
                else:
                    if symb != cur:
                        cur_symb = []
                    cur = symb

                    if len(cur_symb) == 0:
                        temp = self.M - empty - 1
                    else:
                        temp -= 1
                    empty = 0
                    
                    cur_symb.append( i + self.M - 1 )
              
                if (temp <= 0) and cur != "":
                    #u = u + (2**len(cur_symb)) if cur == self.MY_SIGN else u - (2**len(cur_symb))
                    u = u + (per_symb * len(cur_symb)) if cur == self.MY_SIGN else u - (per_symb * len(cur_symb))
                    if len(cur_symb) + 1 == self.M:
                        u = u + len(board) if cur == self.MY_SIGN else u - (len(board)*2)
                    #u = u + (2**len(cur_symb)) if cur == self.MY_SIGN else u - (2**(len(cur_symb) + 2))
                if len(cur_symb) != 0 and cur_symb[0] == i:
                    cur_symb.pop(0)
                cur_i += 1
                cur_j -= 1

        return u * self.GAMMA

def getCornerVal(n, m, row, col):
    l = r = t = b = -1
    l = col - m + 1
    l = l if l >= 0 else 0
    r = col + m - 1
    r = r if r < n else n -1
        
    t = row - m + 1
    t = t if t >= 0 else 0
    b = row + m -1
    b = b if b < n else n -1
    
    return l, r, t, b


# check who win by checking the latest move!
# prev is the sign of agent that makes the move
def winner(board, m, terminate_state, move, prev):
    if terminate_state: # board is full
        return "-"

    n = len(board)
    j, right, i, bottom = getCornerVal(n, m, move[0], move[1])

    # get corner case of diagonals
    temp = m -1 
    
    d1_i = move[0] - temp
    d1_j = move[1] - temp
    if d1_i < 0 or d1_j < 0:
        if d1_i <= d1_j:
            d1_i = 0
            d1_j = move[1] - move[0]
        else:
            d1_i = d1_i - d1_j 
            d1_j = 0

    d2_i = move[0] - temp
    d2_j = move[1] + temp
    if d2_j >= n or d2_i < 0:
        if d2_j >= n:
            d2_i = move[0] + move[1] - n + 1
            d2_j = n - 1
        else:
            d2_i = 0
            d2_j = move[0] + move[1]
    
    diag1 = diag2 = rows = cols = 0 
    row, col = move
    while(i <= bottom or j <= right):
        if j <= right and board[row][j] == prev:
            rows += 1
        else:
            rows = 0
        
        if i <= bottom and board[i][col] == prev:
            cols += 1
        else:
            cols = 0
        
        if d1_i < n and d1_i >= 0 and d1_j >= 0 and d1_j < n and board[d1_i][d1_j] == prev:
            diag1 += 1
        else:
            diag1 = 0

        if d2_i < n and d2_i >= 0 and d2_j < n and d2_j >= 0 and board[d2_i][d2_j] == prev:
            diag2 += 1
        else:
            diag2 = 0
        
        if diag1 == m or diag2 == m or rows == m or cols == m:
            return prev
        i+=1
        j+=1
        d1_i += 1
        d1_j += 1
        d2_i += 1
        d2_j -= 1
    return None
