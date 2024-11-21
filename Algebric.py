import numpy as np
from itertools import combinations

def standardize_problem():
    """
    Takes user input for a linear programming problem and converts it to standard form.
    Handles unrestricted variables.
    Returns the coefficients of the objective function, the constraint matrix, and the RHS vector.
    """
    # Ask if it's a maximization or minimization problem
    optimization_type = input("Do you want to maximize or minimize the objective function? (Enter 'max' or 'min'): ").strip().lower()
    if optimization_type not in ['max', 'min']:
        raise ValueError("Invalid input. Please enter 'max' for maximization or 'min' for minimization.")
    
    # Get the number of variables
    num_variables = int(input("Enter the number of variables in the objective function: "))
    unrestricted = input("Are there any unrestricted variables? (yes/no): ").strip().lower() == "yes"
    
    print(f"Enter the coefficients of the objective function (z = c1*x1 + c2*x2 + ...):")
    c = list(map(float, input("Enter coefficients (space-separated): ").split()))
    if len(c) != num_variables:
        raise ValueError("Number of coefficients does not match the number of variables.")
    
    # Flip the sign of c if it's a minimization problem
    if optimization_type == 'min':
        c = [-coef for coef in c]
    
    # Handle unrestricted variables
    unrestricted_indices = []
    if unrestricted:
        print("Specify which variables are unrestricted (enter indices, e.g., 1 3 for x1 and x3):")
        unrestricted_indices = list(map(int, input("Enter indices (1-based): ").split()))
        unrestricted_indices = [i - 1 for i in unrestricted_indices]  # Convert to 0-based indexing
    
    # Transform unrestricted variables
    new_c = []
    for i in range(num_variables):
        if i in unrestricted_indices:
            # Replace unrestricted variable xi with (xi+ - xi-)
            new_c.append(c[i])   # Coefficient for xi+
            new_c.append(-c[i])  # Coefficient for xi-
        else:
            new_c.append(c[i])
    c = np.array(new_c, dtype=float)

    # Get the constraints
    num_constraints = int(input("Enter the number of constraints: "))
    A = []
    b = []
    inequalities = []
    
    print("For each constraint (e.g., '2*x1 + 3*x2 <= 10'), enter the coefficients and inequality.")
    for i in range(num_constraints):
        print(f"Constraint {i + 1}:")
        row = list(map(float, input("Enter coefficients (space-separated): ").split()))
        if len(row) != num_variables:
            raise ValueError("Number of coefficients does not match the number of variables.")
        inequality = input("Enter inequality (<=, >=, or =): ")
        rhs = float(input("Enter the right-hand side value: "))
        A.append(row)
        b.append(rhs)
        inequalities.append(inequality)
    
    # Transform the constraint matrix for unrestricted variables
    A_transformed = []
    for row in A:
        new_row = []
        for i, coef in enumerate(row):
            if i in unrestricted_indices:
                new_row.append(coef)  # Coefficient for xi+
                new_row.append(-coef)  # Coefficient for xi-
            else:
                new_row.append(coef)
        A_transformed.append(new_row)
    A = np.array(A_transformed, dtype=float)
    b = np.array(b, dtype=float)

    # Handle inequalities
    for i, inequality in enumerate(inequalities):
        if inequality == ">=":
            A[i] *= -1
            b[i] *= -1
        elif inequality == "=":
            # Split equality into two inequalities
            A = np.vstack((A, -A[i]))
            b = np.append(b, -b[i])
    
    # Add slack variables for all <= constraints
    num_slack = len(b)
    slack_matrix = np.eye(num_slack)
    A = np.hstack((A, slack_matrix))
    c = np.append(c, np.zeros(num_slack))  # Add zero coefficients for slack variables
    
    print("\nConverted to standard form:")
    print("Maximize z = ", " + ".join(f"{coef}*x{i+1}" for i, coef in enumerate(c)))
    print("Subject to:")
    for i in range(A.shape[0]):
        print(" + ".join(f"{A[i, j]}*x{j+1}" for j in range(A.shape[1])), "=", b[i])
    print("x >= 0\n")
    
    return c, A, b


def solve_linear_program_algebraic(c, A, b):
    """
    Solve a linear programming problem using the algebraic method.
    Maximize z = c^T * x
    Subject to Ax = b, x >= 0
    """
    num_variables = A.shape[1]
    num_constraints = A.shape[0]

    if num_variables < num_constraints:
        raise ValueError("The problem has fewer variables than constraints, and cannot be solved in standard form.")
    
    basic_indices_combinations = combinations(range(num_variables), num_constraints)
    optimal_value = -np.inf
    optimal_solution = None

    for indices in basic_indices_combinations:
        basic_matrix = A[:, indices]
        
        try:
            basic_solution = np.linalg.solve(basic_matrix, b)
            full_solution = np.zeros(num_variables)
            for idx, value in zip(indices, basic_solution):
                full_solution[idx] = value
            
            if np.all(full_solution >= 0):
                z = c @ full_solution
                if z > optimal_value:
                    optimal_value = z
                    optimal_solution = full_solution
        except np.linalg.LinAlgError:
            continue

    if optimal_solution is None:
        return "No feasible solution exists."
    
    return optimal_value, optimal_solution


# Main Program
def main():
    print("Linear Programming Solver (Algebraic Method)")
    print("-------------------------------------------")
    c, A, b = standardize_problem()
    result = solve_linear_program_algebraic(c, A, b)
    
    if isinstance(result, str):
        print(result)
    else:
        print("\nOptimal Value of z:", result[0])
        print("Optimal Solution x:", result[1])


if __name__ == "__main__":
    main()
