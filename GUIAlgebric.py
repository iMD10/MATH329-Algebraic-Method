import numpy as np
from itertools import combinations
import tkinter as tk
from tkinter import messagebox, ttk

constrints_str = None

def standardize_problem(c, A, b, inequalities, unrestricted_indices):
    num_variables = len(c)
    new_c = []
    for i in range(num_variables):
        if i in unrestricted_indices:
            new_c.append(c[i])
            new_c.append(-c[i])
        else:
            new_c.append(c[i])
    c = np.array(new_c, dtype=float)

    A_transformed = []
    for row in A:
        new_row = []
        for i, coef in enumerate(row):
            if i in unrestricted_indices:
                new_row.append(coef)
                new_row.append(-coef)
            else:
                new_row.append(coef)
        A_transformed.append(new_row)
    A = np.array(A_transformed, dtype=float)
    b = np.array(b, dtype=float)

    new_A = []
    new_b = []
    for i, inequality in enumerate(inequalities):
        if inequality == ">=":
            new_A.append(-A[i])
            new_b.append(-b[i])
        elif inequality == "=":
            new_A.append(A[i])
            new_b.append(b[i])
            new_A.append(-A[i])
            new_b.append(-b[i])
        else:
            new_A.append(A[i])
            new_b.append(b[i])
    A = np.array(new_A, dtype=float)
    b = np.array(new_b, dtype=float)

    num_slack = len(b)
    slack_matrix = np.eye(num_slack)
    A = np.hstack((A, slack_matrix))
    c = np.append(c, np.zeros(num_slack))

    return c, A, b

def solve_linear_program_algebraic(c, A, b):
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

def setwindow(tkobj, w, h):
    ws = tkobj.winfo_screenwidth()
    hs = tkobj.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    tkobj.geometry('%dx%d+%d+%d' % (w, h, x, y))

def variable_selection_page(num_vars, num_constraints):
    var_select_root = tk.Tk()
    var_select_root.title("Select Unrestricted Variables")
    setwindow(var_select_root, 600, 400)

    unrestricted_vars = []

    def finish_selection():
        nonlocal unrestricted_vars
        unrestricted_vars = [i for i, var in enumerate(var_checks) if var.get() == 1]
        var_select_root.destroy()
        coefficient_input_page(num_vars, num_constraints, unrestricted_vars)

    tk.Label(var_select_root, text="Select Unrestricted Variables", font=("Arial", 16)).pack(pady=10)
    var_checks = []
    for i in range(num_vars):
        var = tk.IntVar()
        var_checks.append(var)
        tk.Checkbutton(var_select_root, text=f"x{i + 1}", variable=var).pack(anchor="w", padx=20)

    tk.Button(var_select_root, text="Continue", command=finish_selection).pack(pady=20)
    var_select_root.mainloop()

def coefficient_input_page(num_vars, num_constraints, unrestricted_vars):
    coef_root = tk.Tk()
    coef_root.title("Enter Coefficients")
    setwindow(coef_root, 800, 600)

    obj_coefs = []
    constraint_coefs = []
    rhs_values = []
    inequalities = []

    def next_to_constraints():
        try:
            for i, entry in enumerate(obj_entries):
                obj_coefs.append(float(entry.get()))

            for i in range(num_constraints):
                constraint_row = []
                for j in range(num_vars):
                    constraint_row.append(float(constraint_entries[i][j].get()))
                constraint_coefs.append(constraint_row)

                inequalities.append(ineq_entries[i].get())
                rhs_values.append(float(rhs_entries[i].get()))

            coef_root.destroy()
            solve_problem(obj_coefs, constraint_coefs, rhs_values, inequalities, unrestricted_vars)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Label(coef_root, text="Objective Function Coefficients", font=("Arial", 16)).pack(pady=10)
    obj_entries = []
    for i in range(num_vars):
        frame = tk.Frame(coef_root)
        frame.pack(pady=5)
        tk.Label(frame, text=f"Coefficient for x{i + 1}: ").pack(side="left")
        entry = tk.Entry(frame, width=10)
        entry.pack(side="left")
        obj_entries.append(entry)

    tk.Label(coef_root, text="Constraints", font=("Arial", 16)).pack(pady=10)
    constraint_entries = []
    rhs_entries = []
    ineq_entries = []
    for i in range(num_constraints):
        tk.Label(coef_root, text=f"Constraint {i + 1}").pack(pady=5)
        frame = tk.Frame(coef_root)
        frame.pack(pady=5)

        row_entries = []
        for j in range(num_vars):
            tk.Label(frame, text=f"x{j + 1}:").pack(side="left", padx=5)
            entry = tk.Entry(frame, width=10)
            entry.pack(side="left", padx=5)
            row_entries.append(entry)
        constraint_entries.append(row_entries)

        tk.Label(frame, text="Inequality:").pack(side="left", padx=5)
        ineq_entry = ttk.Combobox(frame, values=["<=", ">=", "="], width=5)
        ineq_entry.set("<=")
        ineq_entry.pack(side="left", padx=5)
        ineq_entries.append(ineq_entry)

        tk.Label(frame, text="RHS:").pack(side="left", padx=5)
        rhs_entry = tk.Entry(frame, width=10)
        rhs_entry.pack(side="left", padx=5)
        rhs_entries.append(rhs_entry)

    tk.Button(coef_root, text="Solve", command=next_to_constraints).pack(pady=20)
    coef_root.mainloop()

def solve_problem(obj_coefs, constraint_coefs, rhs_values, inequalities, unrestricted_vars):
    try:
        global opt_type
        if opt_type == "min":
            obj_coefs = [-coef for coef in obj_coefs]

        c, A, b = standardize_problem(obj_coefs, constraint_coefs, rhs_values, inequalities, unrestricted_vars)
        result = solve_linear_program_algebraic(c, A, b)
        if isinstance(result, str):
            msg = result
        else:
            z, x = result
            if opt_type == "min":
                z = -z
            msg = f"Optimal Value (z): {z}\nOptimal Solution (x): {x.tolist()}"

        result_root = tk.Tk()
        result_root.title("Result")
        setwindow(result_root, 400, 200)
        tk.Label(result_root, text=msg, font=("Arial", 12)).pack(pady=20)

        def program_loop(choice):
            result_root.destroy()
            if choice:
                main()
            else:
                exit()

        tk.Button(result_root, text="Solve Another", command=lambda: program_loop(True)).pack(pady=10)
        tk.Button(result_root, text="Exit", command=lambda: program_loop(False)).pack(pady=10)
        result_root.mainloop()

    except Exception as e:
        messagebox.showerror("Error", str(e))

def gui_solver():
    root = tk.Tk()
    root.title("Linear Programming Solver - Define Problem")
    setwindow(root, 600, 400)

    def next_to_variable_selection():
        try:
            global opt_type
            opt_type = opt_type_var.get()
            num_vars = int(num_vars_entry.get())
            num_constraints = int(num_constraints_entry.get())
            if num_vars <= 0 or num_constraints <= 0:
                raise ValueError("Number of variables and constraints must be greater than zero.")
            root.destroy()
            variable_selection_page(num_vars, num_constraints)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Label(root, text="Define Your Linear Programming Problem", font=("Arial", 16)).pack(pady=10)
    tk.Label(root, text="Is this a maximization or minimization problem?").pack(pady=5)
    opt_type_var = tk.StringVar(value="max")
    max_button = tk.Radiobutton(root, text="Maximization", variable=opt_type_var, value="max")
    max_button.pack()
    min_button = tk.Radiobutton(root, text="Minimization", variable=opt_type_var, value="min")
    min_button.pack()

    tk.Label(root, text="Enter the number of variables:").pack(pady=5)
    num_vars_entry = tk.Entry(root, width=10)
    num_vars_entry.pack()

    tk.Label(root, text="Enter the number of constraints:").pack(pady=5)
    num_constraints_entry = tk.Entry(root, width=10)
    num_constraints_entry.pack()

    tk.Button(root, text="Next", command=next_to_variable_selection).pack(pady=20)
    root.mainloop()

def main():
    root = tk.Tk()
    root.title("Linear Programming Solver")
    setwindow(root, 400, 300)

    def start_solver():
        root.destroy()
        gui_solver()

    tk.Label(root, text="Welcome to Linear Programming Solver!", font=("Arial", 16)).pack(pady=20)
    tk.Button(root, text="Start Solver", font=("Arial", 14), command=start_solver, width=20).pack(pady=10)
    tk.Button(root, text="Exit", font=("Arial", 14), command=root.destroy, width=20).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main()
