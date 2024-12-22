use ndarray::prelude::*;
use ndarray::Array1;

pub fn simplex_solver(
    c: Array1<f64>,     //objective function coeffs
    a: &Array2<f64>,    //constraint coeffs
    b: &Array1<f64>,    //RHS values
) -> Option<(Array1<f64>, f64)> {
    let num_constraints = a.nrows();
    let num_vars = a.ncols();
    let mut tableau = Array2::<f64>::zeros((num_constraints + 1, num_vars + 1));

    //tableau init
    tableau.slice_mut(s![..-1, ..num_vars]).assign(a);
    tableau.slice_mut(s![..-1, -1]).assign(b);
    tableau.slice_mut(s![-1, ..num_vars]).assign(&(-&c));

    loop {
        let last_row_index = tableau.nrows() - 1;
        let last_col_index = tableau.ncols() - 1;

        //check if optimal solution
        if tableau.slice(s![last_row_index, 0..last_col_index]).iter().all(|&val| val >= 0.0) {
            break;
        }

        //find the pivot column idx
        let pivot_col = find_pivot_column(&tableau, last_row_index)?;

        //find the pivot row
        if let Some(pivot_row) = find_pivot_row(&tableau, pivot_col, last_row_index) {
            //pivot
            pivot_operation(&mut tableau, pivot_row, pivot_col);
        } else {
            eprintln!("Problem is unbounded: no valid leaving variable.");
            return None;
        }
    }

    extract_solution(&tableau)
}

fn find_pivot_column(tableau: &Array2<f64>, last_row_index: usize) -> Option<usize> {
    tableau
        .slice(s![last_row_index, ..]) //take the last row (objective coeffs)
        .iter().enumerate() //make it into (index, val) tuples array
        .filter(|&(_, &val)| val < 0.0)//take only the negative values
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) //find the lowest value of them (partial cmp because there might only be one negative number)
        .map(|(idx, _)| idx) //if there is no negative values return None, otherwise return the index of that val (|idx, _| matches the val from min_by into idx) 
}

fn find_pivot_row(tableau: &Array2<f64>, pivot_col: usize, last_row_index: usize) -> Option<usize> {
    tableau
        .slice(s![..last_row_index, pivot_col])//takes all rows except the last one (constraint coeffs) and only take those from the previously found col index
        .iter().enumerate()//matches them into (idx, val)
        .filter(|&(_, &val)| val > 0.0)//takes only positive values
        .map(|(row, &val)| (row, tableau[[row, tableau.ncols() - 1]] / val)) //keep the idx and change the val to the ratio rhs/col val (tableau[[row, tableau.ncols() - 1]] takes the value in the last column of the row)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) //take the lowest val of them
        .map(|(row, _)| row) //return the idx
}

fn pivot_operation(tableau: &mut Array2<f64>, pivot_row: usize, pivot_col: usize) {
    let pivot_value = tableau[[pivot_row, pivot_col]];
    if pivot_value == 0.0 {
        panic!("Pivot value is zero, cannot divide.");
    }
    //divide each val in the pivot row by the found pivot val
    tableau.row_mut(pivot_row).map_inplace(|x| *x /= pivot_value);

    for row in 0..tableau.nrows() {
        if row != pivot_row {
            let row_factor = tableau[[row, pivot_col]]; //how much of the pivot row needs to be subtracted from the current row to make the pivot column entry zero
            let pivot_row_copy = tableau.row(pivot_row).to_owned(); //get a coppy of the pivot row 
            tableau
                .row_mut(row) //get a mutable reference of the row 
                .zip_mut_with(&pivot_row_copy, |x, &y| *x -= row_factor * y); //subtracts row_factor*pivot row val from the current row
        }
    }
}

//extract solution and objective value from the tabeau
fn extract_solution(tableau: &Array2<f64>) -> Option<(Array1<f64>, f64)> {
    let last_row_index = tableau.nrows() - 1;
    let last_col_index = tableau.ncols() - 1;

    let mut solution = Array1::<f64>::zeros(tableau.ncols() - 1);
    let basis = find_basis(tableau);
    for &col in &basis {
        if col < solution.len() {
            solution[col] = tableau[[basis.iter().position(|&x| x == col).unwrap(), last_col_index]];
        }
    }

    let objective_value = tableau[[last_row_index, last_col_index]];
    Some((solution, objective_value))
}

//find basic variables
fn find_basis(tableau: &Array2<f64>) -> Vec<usize> {
    let mut basis = Vec::new();
    for col in 0..tableau.ncols() - 1 {
        let column = tableau.slice(s![..-1, col]);
        if column.iter().filter(|&&x| x != 0.0).count() == 1 && column.iter().sum::<f64>() == 1.0 {
            basis.push(col);
        }
    }
    basis
}

fn print_solution(solution: &Array1<f64>) {
    // Assuming the solution has at least 6 variables
    let x_vals = &solution.slice(s![..solution.len() - 3]); // All variables except the last 3
    let s_vals = &solution.slice(s![solution.len() - 3..]); // Last 3 variables (slack variables)

    let formatted_x_vals = x_vals.iter()
        .enumerate()
        .map(|(i, &val)| format!("x{}: {}", i + 1, val))
        .collect::<Vec<String>>()
        .join(", ");

    let formatted_s_vals = s_vals.iter()
        .enumerate()
        .map(|(i, &val)| format!("s{}: {}", i + 1, val))
        .collect::<Vec<String>>()
        .join(", ");

    println!("x vals: [{}] s vals: [{}]", formatted_x_vals, formatted_s_vals);
}

fn main() {
    let c = array![2.0, -3.0, 0.0, -5.0, 0.0, 0.0, 0.0];
    let a = array![
        [-1.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0],
        [2.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    ];
    let b = array![8.0, 10.0, 3.0];

    if let Some((solution, objective_value)) = simplex_solver(c.clone(), &a, &b) {
        print_solution(&solution);
        println!("Optimal objective value: {}", objective_value);
        println!("Base (indices of basic variables): {:?}", find_basis(&a));
    } else {
        println!("The problem is unbounded or infeasible.");
    }

    let n = array![8.0, 0.0, 3.0];
    if let Some((solution, objective_value)) = simplex_solver(c, &a, &n) {
        print_solution(&solution);
        println!("Optimal objective value: {}", objective_value);
        println!("Base (indices of basic variables): {:?}", find_basis(&a));
    } else {
        println!("The problem is unbounded or infeasible.");
    }
}
