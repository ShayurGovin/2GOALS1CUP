import numpy as np

class Munkres:
    def __init__(self):
        self.C = None
        self.row_covered = None
        self.col_covered = None
        self.n = None
        self.Z0_r = None
        self.Z0_c = None
        self.marked = None
        self.path = None

    def pad_matrix(self, matrix):
        max_columns = max(matrix.shape[1], matrix.shape[0])
        new_matrix = np.zeros((max_columns, max_columns))
        new_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        return new_matrix

    def compute(self, cost_matrix):
        self.C = self.pad_matrix(np.copy(cost_matrix))
        self.n = self.C.shape[0]
        self.original_c = np.copy(self.C)
        self.row_covered = np.zeros(self.n, dtype=bool)
        self.col_covered = np.zeros(self.n, dtype=bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((self.n * 2, 2), dtype=int)
        self.marked = np.zeros((self.n, self.n), dtype=int)

        done = False
        step = 1

        steps = { 1 : self.step1,
                  2 : self.step2,
                  3 : self.step3,
                  4 : self.step4,
                  5 : self.step5,
                  6 : self.step6 }

        while not done:
            try:
                func = steps[step]
                step = func()
                if step == 7:
                    done = True
            except KeyError:
                done = True

        results = []
        for i in range(self.original_c.shape[0]):
            for j in range(self.original_c.shape[1]):
                if self.marked[i][j] == 1:
                    results.append((i, j))

        return results

    def step1(self):
        self.C -= self.C.min(axis=1)[:, np.newaxis]
        for i in range(self.n):
            for j in range(self.n):
                if self.C[i][j] == 0 and not self.col_covered[j] and not self.row_covered[i]:
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True
                    break
        self.row_covered[:] = False
        self.col_covered[:] = False
        return 2

    def step2(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 1:
                    self.col_covered[j] = True
        count = np.sum(self.col_covered)
        if count >= self.n:
            return 7
        else:
            return 3

    def step3(self):
        while True:
            r, c = self.find_a_zero()
            if r == -1:
                return 5
            self.marked[r][c] = 2
            if self.find_star_in_row(r) >= 0:
                col = self.find_star_in_row(r)
                self.row_covered[r] = True
                self.col_covered[col] = False
            else:
                self.Z0_r = r
                self.Z0_c = c
                return 4

    def step4(self):
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        while True:
            row = self.find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                break
            col = self.find_prime_in_row(path[count][0])
            count += 1
            path[count][0] = path[count-1][0]
            path[count][1] = col

        self.convert_path(path, count)
        self.clear_covers()
        self.erase_primes()
        return 2

    def step5(self):
        minval = np.inf
        for i in range(self.n):
            for j in range(self.n):
                if not self.row_covered[i] and not self.col_covered[j]:
                    if minval > self.C[i][j]:
                        minval = self.C[i][j]
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i][j] += minval
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
        return 3

    def step6(self):
        return 7

    def find_a_zero(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.C[i][j] == 0 and not self.row_covered[i] and not self.col_covered[j]:
                    return i, j
        return -1, -1

    def find_star_in_row(self, row):
        for j in range(self.n):
            if self.marked[row][j] == 1:
                return j
        return -1

    def find_star_in_col(self, col):
        for i in range(self.n):
            if self.marked[i][col] == 1:
                return i
        return -1

    def find_prime_in_row(self, row):
        for j in range(self.n):
            if self.marked[row][j] == 2:
                return j
        return -1

    def convert_path(self, path, count):
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def clear_covers(self):
        self.row_covered[:] = False
        self.col_covered[:] = False

    def erase_primes(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

def role_assignment(teammate_positions, formation_positions):

    teammates = np.array(teammate_positions)
    formations = np.array(formation_positions)

    cost_matrix = np.linalg.norm(teammates[:, np.newaxis] - formations, axis=2)

    m = Munkres()
    indexes = m.compute(cost_matrix)

    point_preferences = {i + 1: formations[j] for i, j in indexes}

    return point_preferences


def pass_reciever_selector(player_unum, teammate_positions, final_target):
    
    # Input : Locations of all teammates and a final target you wish the ball to finish at
    # Output : Target Location in 2d of the player who is recieveing the ball
    #-----------------------------------------------------------#

    # Example
    pass_reciever_unum = player_unum + 1                  #This starts indexing at 1, therefore player 1 wants to pass to player 2
    
    if pass_reciever_unum != 12:
        target = teammate_positions[pass_reciever_unum-1] #This is 0 indexed so we actually need to minus 1 
    else:
        target = final_target 
    
    return target