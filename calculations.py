import numpy as np


def get_fundamental_subspaces(A, tol=1e-10):
    A = np.array(A, dtype=float)
    m, n = A.shape
    U, S, Vh = np.linalg.svd(A)

    # Determine rank
    rank = np.sum(S > tol)

    return {
        "column_space": U[:, :rank],
        "left_null_space": U[:, rank:],
        "row_space": Vh[:rank, :].T,
        "null_space": Vh[rank:, :].T,
        "rank": rank
    }


def manual_test():
    print("--- Matrix Subspace Tester ---")
    try:
        rows = int(input("Enter number of rows (max 5): "))
        cols = int(input("Enter number of columns (max 5): "))

        if rows > 5 or cols > 5:
            print("Keep it 5x5 or smaller as requested!")
            return

        print(f"Enter the {rows}x{cols} matrix values row by row (space-separated):")
        grid = []
        for i in range(rows):
            row_input = input(f"Row {i + 1}: ").split()
            if len(row_input) != cols:
                print(f"Error: Expected {cols} values.")
                return
            grid.append([float(x) for x in row_input])

        A = np.array(grid)
        results = get_fundamental_subspaces(A)

        print("\n--- RESULTS ---")
        print(f"Matrix Rank: {results['rank']}")

        for name, space in results.items():
            if name == "rank": continue
            print(f"\n{name.replace('_', ' ').title()} Basis:")
            print(space if space.size > 0 else "Empty (Zero Vector Space)")

        # Validation Check: A * Null_Space should = 0
        if results['null_space'].size > 0:
            verification = np.allclose(A @ results['null_space'], 0, atol=1e-10)
            print(f"\nValidation (A * NullSpace == 0): {verification}")

    except ValueError:
        print("Invalid input. Please enter numbers only.")


if __name__ == "__main__":
    manual_test()
