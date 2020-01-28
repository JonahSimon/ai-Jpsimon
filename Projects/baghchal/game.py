from typing import List, Tuple
from const import Const
from move import Move

class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self._board : List[List[int]] = [[Const.MARK_NONE for col in range(Const.COLS)] for row in range(Const.ROWS)]
        self._board[0][0] = Const.MARK_TIGER
        self._board[Const.ROWS-1][0] = Const.MARK_TIGER
        self._board[0][Const.COLS-1] = Const.MARK_TIGER
        self._board[Const.ROWS-1][Const.COLS-1] = Const.MARK_TIGER
        self._state : int = Const.STATE_TURN_GOAT
        self._placed : int = 0 # number of goats placed
        self._captured : int = 0 # number of goats captured
        self._unplayed = Const.ROWS*Const.COLS - 4

    @property
    def over(self) -> bool:
        return \
            self._state == Const.STATE_WIN_GOAT or \
            self._state == Const.STATE_WIN_TIGER or \
            self._state == Const.STATE_DRAW

    def moveOk(self,move : Move) -> None:
        if move.mark == Const.MARK_TIGER and self._state != Const.STATE_TURN_TIGER:
            raise ValueError("tiger cannot play")
        if move.mark == Const.MARK_GOAT and self._state != Const.STATE_TURN_GOAT:
            raise ValueError("goat cannot play")
        if self._board[move.toRow][move.toCol] != Const.MARK_NONE:
            raise ValueError("destination (to) is occupied") 
        if not move.placement and self._board[move.fromRow][move.fromCol] != move.mark:
            raise ValueError("source (from) is not player")
        if move.capture:
            captureRow : int = (move.toRow + move.fromRow) // 2
            captureCol : int = (move.toCol + move.fromCol) // 2
            if self._board[captureRow][captureCol] != Const.MARK_GOAT:
                raise ValueError("capture move without goat")


    def getMarkedPlaces(self,mark : int,row : int ,col : int,dist : int) -> List[Tuple[int,int]]:
        places : List[Tuple[int,int]]
        r : int =row-dist
        c : int =col-dist
        for dd in [(1,0),(0,1),(-1,0),(0,-1)]:
            for d in range(dist):
                if 0 <= r and r < Const.ROWS and \
                        0 <= c and c <= Const.COLS and \
                        self._board[r][c] == mark:
                    places.append((r,c))
                r += dd[0]
                c += dd[1]
        return places

    def goatPlaceMoves(self,mark : int):
        ans : List[Move] = List[Move]
              if self._placed < Const.GOAT_PLACEMENTS:
            for row in range(Const.ROWS):
                for col in range(Const.COLS):
                    if self._board[row][col] == mark:
                        offsets : List[Tuple[int,int]]= self.getMarkedPlaces(mark,row,col,1)
                        for offset in offsets:
                            move = Move(mark,row,col,offset[0],offset[1])
                            moves.append(move)

                        for drow in range(-1,2):
                            for dcol in range(-1,2):



    @property
    def moves(self) -> List[Move]:
        if self.over:
            return []
        if self._state == Const.STATE_TURN_TIGER:
            return self.tigerMoves()
        else:
            return self.goatMoves()

        mark = None
        if self._state == Const.STATE_TURN_O:
            mark = Const.MARK_O
        elif self._state == Const.STATE_TURN_X:
            mark = Const.MARK_X

        if mark == None:
            return []

        moves = []
        for row in range(Const.ROWS):
            for col in range(Const.COLS):
                if self._board[row][col] == Const.MARK_NONE:
                    moves.append(Move(row,col,mark))

        return moves

    def _repeats(self,row,col,rowDir,colDir):
        mark = self._board[row][col]        
        count = 0
        while (row >= 0 and row < Const.ROWS and col >= 0) and \
              col < Const.COLS and self._board[row][col]==mark:
            row=row+rowDir
            col=col+colDir
            count = count + 1
        return count

    def _length(self,row,col,rowDir,colDir):
        return self._repeats(row,col,rowDir,colDir)+self._repeats(row,col,-rowDir,-colDir)-1

    def _winRow(self,row,col):
        return self._length(row,col,1,0) >= Const.WIN_LENGTH
    def _winCol(self,row,col):
        return self._length(row,col,0,1) >= Const.WIN_LENGTH
    def _winMainDiag(self,row,col):
        return self._length(row,col,1,1) >= Const.WIN_LENGTH
    def _winOffDiag(self,row,col):
        return self._length(row,col,1,-1) >= Const.WIN_LENGTH

    def _win(self,row,col):
        return self._winRow(row,col) or self._winCol(row,col) or \
               self._winMainDiag(row,col) or self._winOffDiag(row,col)

    def _draw(self):
        return self._unplayed == 0

    def move(self,move : Move):
        self.moveOk(row,col,mark)
        self._board[row][col]=mark
        self._unplayed = self._unplayed - 1
        if self._win(row,col):
            if mark == Const.MARK_O:
                self._state = Const.STATE_WIN_O
            if mark == Const.MARK_X:
                self._state = Const.STATE_WIN_X
        elif self._draw():
            self._state = Const.STATE_DRAW
        else:
            if mark == Const.MARK_O:
                self._state = Const.STATE_TURN_X
            if mark == Const.MARK_X:
                self._state = Const.STATE_TURN_O
    def unplay(self, game):
        if self.place:
            game.goats = game.goats - 1
            game.state = Const.STATE_TURN_GOAT
            game.board[self._toRow][self._toCol] = Const.MARK_NONE
            game.unplayed = game.unplayed + 1
        else:
            game.state = Const.STATE_TURN_GOAT if self._mark == Const.MARK_GOAT else Const.STATE_TURN_TIGER

    def unmove(self,row,col):
        Const.rowOk(row)
        Const.colOk(col)
        if self._board[row][col] == Const.MARK_X:
            self._unplayed = self._unplayed + 1
            self._board[row][col] = Const.MARK_NONE
            self._state = Const.STATE_TURN_X
        elif self._board[row][col] == Const.MARK_O:
            self._unplayed = self._unplayed + 1
            self._board[row][col] = Const.MARK_NONE
            self._state = Const.STATE_TURN_O
        else:
            raise ValueError("unmove (" + str(row) + "," + str(col) + ") invalid in current state")

    def getBoard(self):
        return [[self._board[row][col]  for col in range(Const.COLS)] for row in range(Const.ROWS)]
        
    def getState(self):
        return self._state
    
    
    def play(self,moves):
        for move in moves.split():
            Move.parse(move).play(self)

    def __str__(self):
        ans = "\n"
        for row in range(Const.ROWS):
            s="";
            for col in range(Const.COLS):
                s=s+Const.markStr(self._board[row][col])
            ans = ans + s + "\n"
        return ans

    def copyTo(self,target):
        target._board = [[self._board[row][col] for col in range(Const.COLS)] for row in range(Const.ROWS)]
        target._state = self._state
        target._unplayed = self._unplayed

    def clone(self):
        ans = Game()
        self.copyTo(ans)
        return ans

    def __hash__(self):
        tuple = (self._board[k % Const.ROWS][k // Const.ROWS] for k in range(Const.ROWS*Const.COLS))
        return hash(tuple)

    def __eq__(self, other):
        return other != None and (self._board == other._board)

    def __ne__(self, other):
        return not self.__eq__(other)

    def getIndex(self):
        i = 0
        for row in range(Const.ROWS):
            for col in range(Const.COLS):
                board=self._board[row][col]
                x = 0
                if board == Const.MARK_X:
                    x = 1
                elif board == Const.MARK_O:
                    x = 2
                i = 3*i + x
        return i

    def flipRows(self):
        ans=self.clone()
        for row in range(Const.ROWS):
            for col in range(Const.COLS):
                ans._board[Const.ROWS-row-1][col]=self._board[row][col]
        return ans

    def flipCols(self):
        ans = self.clone()
        for row in range(Const.ROWS):
            for col in range(Const.COLS):
                ans._board[row][Const.COLS-col-1]=self._board[row][col]
        return ans

    def flipRowsAndCols(self):
        ans = self.clone()
        for row in range(Const.ROWS):
            for col in range(Const.COLS):
                ans._board[Const.ROWS-row-1][Const.COLS-col-1]=self._board[row][col]
        return ans

    def transpose(self):
        if Const.ROWS != Const.COLS: return None
        ans = self.clone()
        for row in range(Const.ROWS):
            for col in range(Const.COLS):
                ans._board[col][row]=self._board[row][col]
        return ans

    def offDiagonalTranspose(self):
        if Const.ROWS != Const.COLS: return None        
        return self.flipRows().transpose()
    
    def getEquivClass(self):
        return [self.clone(),self.flipRows(),self.flipCols(),self.flipRowsAndCols()]
        
        
