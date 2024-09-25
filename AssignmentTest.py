# The Implementation of the Hungarian class below is its own personalised implementation but a few online resources helped in deriving it:
# https://github.com/Ibrahim5aad/kuhn-munkres-algorithm/blob/master/hungarian_method.py
# https://github.com/tdedecko/hungarian-algorithm/blob/master/hungarian.py
# https://plainenglish.io/blog/hungarian-algorithm-introduction-python-implementation-93e7c0890e15

import numpy as np

class Hungarian:
    #  Implementation of the Hungarian algorithm (also known as the Munkres algorithm)
    def __init__(self):
        # Initialize variables used throughout the algorithm
        self.C = None  # Cost matrix
        self.row_covered = None  # Boolean array to track covered rows
        self.col_covered = None  # Boolean array to track covered columns
        self.n = None  # Size of the square matrix
        self.Z0_r = None  # Temporary storage for row index
        self.Z0_c = None  # Temporary storage for column index
        self.marked = None  # Matrix to mark zeros
        self.path = None  # Path of alternating zeros

    def pad_matrix(self, matrix):
        # Pad the input matrix with zeros to make it square if necessary.
        max_columns = max(matrix.shape[1], matrix.shape[0])
        new_matrix = np.zeros((max_columns, max_columns))
        new_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        return new_matrix

    def compute(self, cost_matrix):
        # Initialize the algorithm with the given cost matrix
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

        # Define the steps of the algorithm
        steps = { 1 : self.step1,
                  2 : self.step2,
                  3 : self.step3,
                  4 : self.step4,
                  5 : self.step5,
                  6 : self.step6 }

        # Main loop of the algorithm
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
        # Find the first uncovered element with value 0
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
        # Find the first starred element in the specified column
        for i in range(self.n):
            if self.marked[i][col] == 1:
                return i
        return -1

    def find_prime_in_row(self, row):
        # Find the first prime element in the specified row
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
        # Clear all covered matrix cells
        self.row_covered[:] = False
        self.col_covered[:] = False

    def erase_primes(self):
        # Erase all prime markings
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

def role_assignment(teammate_positions, formation_positions):
    # Assign roles to teammates based on their positions and desired formation positions.
    teammates = np.array(teammate_positions)
    formations = np.array(formation_positions)

    # Calculate the cost matrix based on Euclidean distances
    cost_matrix = np.linalg.norm(teammates[:, np.newaxis] - formations, axis=2)

    # Use the Hungarian algorithm to find the optimal assignment
    Hung = Hungarian()
    indexes = Hung.compute(cost_matrix)

    # Create a dictionary of assignments
    point_preferences = {i + 1: formations[j] for i, j in indexes}

    return point_preferences

def test_role_assignment():
    # Provided teammate positions and formation positions
    teammate_positions = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])
    
    formation_positions = [
        [-13, 0], 
        [-10, -2], 
        [-11, 3], 
        [-8, 0], 
        [-3, 0], 
        [0, 1], 
        [2, 0], 
        [3, 3],
        [8, 0], 
        [9, 1], 
        [12, 0]
    ]

    # Call the function and print the result
    result = role_assignment(teammate_positions, formation_positions)
    print("Optimal Assignments:", result)

# Run the test function when the script is executed
if __name__ == "__main__":
    test_role_assignment()